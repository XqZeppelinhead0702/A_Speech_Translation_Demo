"""Realtime duplex transport; ticket issued over the authenticated HTTP start route."""
import asyncio
import hmac
import inspect
import json
import logging
import struct
import time

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState


def register_socket(api, service, tickets, owner, prefix):
    @api.websocket(prefix + "/stream")
    async def stream(ws: WebSocket):
        tasks = []
        session_id = None
        started = False
        send_lock = asyncio.Lock()

        async def send(payload):
            async with send_lock:
                payload["sent_server_time_ms"] = time.time() * 1000
                await ws.send_json(payload)

        try:
            owner_id = owner(ws)
            if inspect.isawaitable(owner_id):
                owner_id = await owner_id
            await ws.accept()
            hello = await asyncio.wait_for(ws.receive_json(), timeout=10)
            asr_updates = hello.get("asr_updates", False)
            if type(asr_updates) is not bool:
                raise ValueError("ASR 推送选项无效。")
            session_id = hello.get("session_id")
            ticket = tickets.get(session_id)
            if not ticket or ticket[1] != owner_id or ticket[2] < time.monotonic() or not hmac.compare_digest(
                    str(hello.get("ticket", "")), ticket[0]):
                raise ValueError("实时连接授权无效，请重新开始测试。")
            service.poll_results(session_id, owner_id=owner_id)
            del tickets[session_id]
            await send({"type": "ready"})

            async def receive_audio():
                nonlocal started
                while True:
                    message = await ws.receive()
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect(message.get("code", 1000))
                    packet = message.get("bytes")
                    if packet is not None:
                        if not 10 <= len(packet) <= 2 * 1024 * 1024 or (len(packet) - 8) % 2:
                            raise ValueError("无效的 PCM16 实时音频包。")
                        sequence, rate = struct.unpack("<II", packet[:8])
                        ack = service.push_audio(session_id, owner_id=owner_id, chunk_seq=sequence,
                                                 sample_rate=rate, audio=np.frombuffer(packet[8:], dtype="<i2"))
                        started = True
                        await send({"type": "ack", **ack})
                    else:
                        text = message.get("text", "")
                        if len(text) > 4096:
                            raise ValueError("实时控制消息过大。")
                        control = json.loads(text)
                        if control.get("type") == "ping":
                            await send({"type": "pong", "client_time_ms": control.get("client_time_ms")})
                        elif control.get("type") == "finish":
                            service.finish_session(session_id, last_chunk_seq=control.get("last_chunk_seq"), owner_id=owner_id)
                        else:
                            raise ValueError("无效的实时控制消息。")

            async def publish_results():
                cursor = 0
                last_draft_seq = -1
                last_asr = None
                last_mt_version = -1
                last_send = 0.
                while True:
                    result = service.poll_results(session_id, cursor=cursor, owner_id=owner_id)
                    now = time.monotonic()
                    terminal = result["status"] in {"finished", "error", "cancelled"}
                    snapshot = result["snapshot"]
                    asr = (snapshot["asr_version"], snapshot["asr_final"])
                    if asr_updates and asr != last_asr:
                        # Send ASR first, without translation strings/history or an
                        # MT refresh interval. Translation may legitimately lag.
                        await send({"type": "asr", "snapshot": {key: snapshot.get(key) for key in
                            ["transcript", "asr_version", "asr_final", "asr_server_time_ms",
                             "first_asr_seconds", "first_audio_seconds"]}})
                        last_asr = asr
                    draft_seq = snapshot.get("translation_draft_seq", 0)
                    mt_version = snapshot["translation_version"]
                    mt_changed = draft_seq != last_draft_seq or mt_version != last_mt_version
                    # Larger MT/history packets are paced; audio ACKs and small
                    # ASR packets are not throttled by this interval.
                    legacy_update = not asr_updates and (result["cursor"] != cursor or mt_changed)
                    if terminal or legacy_update or (mt_changed and now-last_send >= .1) or now-last_send >= .5:
                        last_draft_seq = draft_seq
                        last_mt_version = mt_version
                        cursor, last_send = result["cursor"], now
                        await send({"type": "results", "result": result})
                    if terminal:
                        return
                    await asyncio.sleep(.02 if asr_updates else .05)

            tasks = [asyncio.create_task(receive_audio()), asyncio.create_task(publish_results())]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except WebSocketDisconnect as exc:
            logging.info("Realtime socket %s disconnected: code=%s audio_started=%s",
                         session_id, exc.code, started)
        except RuntimeError:
            pass
        except Exception as exc:
            logging.warning("Realtime socket %s failed: %s", session_id, exc)
            try:
                await send({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            # A handshake-only failure must leave the session available to HTTP fallback.
            if session_id and started:
                try:
                    if service.poll_results(session_id, owner_id=owner_id)["status"] in {"listening", "draining"}:
                        service.cancel_session(session_id, owner_id=owner_id)
                except ValueError:
                    pass
            try:
                # Receiving a disconnect does not necessarily update send state.
                # Avoid another frame to a known-dead peer, but still handle a
                # disconnect racing with close after the state checks.
                if (ws.client_state != WebSocketState.DISCONNECTED
                        and ws.application_state != WebSocketState.DISCONNECTED):
                    await ws.close()
            except (WebSocketDisconnect, OSError, RuntimeError):
                pass
