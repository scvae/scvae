import re

from pygments.lexers.shell import BashSessionLexer
from pygments.token import Number

class TerminalLexer(BashSessionLexer):
    name = "terminal"

    def get_tokens_unprocessed(self, text):
        for index, token, value in BashSessionLexer.get_tokens_unprocessed(
                self, text):
            if re.match(r"\.\d+", value):
                token = Number
            yield index, token, value
