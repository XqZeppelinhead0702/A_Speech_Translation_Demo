"""Readable subtitle projection, independent of a translator's decoder position.

A fresh full-text decoder starts at character zero. Its short prefixes are not
retractions of previously visible subtitles. Preserve those subtitles until the
candidate extends them or a validated complete hypothesis replaces them.
"""
import time


def common_prefix_length(left, right):
    index = 0
    for first, second in zip(left, right):
        if first != second:
            break
        index += 1
    return index


class SubtitleProjection:
    def __init__(self):
        self.text, self.source, self.revision = "", "", 0
        self.provisional = False
        self.confirmed_text, self.confirmed_source, self.confirmed_revision = "", "", 0
        self.sequence, self.server_time_ms = 0, None

    def _set(self, text, revision, source, provisional):
        value = (text, revision, source, provisional)
        if value == (self.text, self.revision, self.source, self.provisional):
            return False
        self.text, self.revision, self.source, self.provisional = value
        self.sequence += 1
        self.server_time_ms = round(time.time()*1000, 3)
        return True

    def partial(self, candidate, revision, source):
        if not candidate or revision < self.revision or candidate == self.text:
            return False
        # Prefix replay (including its temporary ending punctuation) never makes
        # the displayed text shrink or switches it to a newer source version.
        if len(candidate) < len(self.text):
            return False
        if self.text and not candidate.startswith(self.text):
            prefix = common_prefix_length(self.text, candidate)
            # A small editable suffix can change once the candidate has caught up.
            # This is a presentation guard, not a semantic/permanent commit rule.
            tail = max(2, min(24, len(self.text)//5))
            if prefix < max(1, len(self.text)-tail):
                return False
        return self._set(candidate, revision, source, True)

    def complete(self, candidate, revision, source):
        if revision < self.revision:
            return False
        # Only the service's successfully validated return value may come here.
        # Complete hypotheses can fix earlier wording or be shorter; never freeze
        # an old mistranslation merely to make subtitles monotonically increase.
        self.confirmed_text, self.confirmed_revision, self.confirmed_source = candidate, revision, source
        return self._set(candidate, revision, source, False)

    def rollback(self):
        return self._set(self.confirmed_text, self.confirmed_revision, self.confirmed_source, False)

    def snapshot(self):
        return dict(translation_display=self.text, translation_display_version=self.revision,
                    translation_display_source=self.source, translation_display_provisional=self.provisional,
                    translation_display_seq=self.sequence, translation_display_server_time_ms=self.server_time_ms)
