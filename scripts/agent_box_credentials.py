"""Secure capture and storage of the ESnet AmSC token (Task 3).

Single persisted location, relative to a supplied home:
``agent-box/credentials/esnet/amsc-token`` with every created directory
held at mode 0700 and the token file at mode 0600.

Rules enforced here:

- Writes are atomic in the destination directory (mkstemp + fsync +
  replace + directory fsync).  A failed replace preserves the previous
  token and removes the temp file; the underlying OSError propagates so
  callers see the real failure.
- Exactly one terminal line ending (CRLF or LF) is removed from a
  captured value; all other content, including leading/trailing
  whitespace, is preserved verbatim.  Empty values are rejected.
- Protected reads ``lstat`` the candidate and reject symlinks,
  non-regular files, files owned by another user, and any file whose
  mode grants group or other access — before the file is opened.
- Interactive capture uses a hidden reader (``getpass.getpass`` by
  default) and never writes the secret to any output stream, prompt,
  exception, log, argument, or environment variable.
- ``token_status`` reports only ``"absent"``, ``"present"`` or
  ``"unsafe"`` — it never returns or displays token content.

The actual network validation client ships in a later task; callers
that need validation must supply a validator callback (the setup
wizard fails closed with exit 77 without one).
"""
import getpass
import os
import stat
import tempfile

# Relative location of the persisted AMSC token under the supplied home.
AMSC_TOKEN_RELATIVE_PATH = os.path.join(
    "agent-box", "credentials", "esnet", "amsc-token"
)

# Private mode for every directory created along the credential path.
_PRIVATE_DIR_MODE = 0o700

# Private mode for the token file itself.
_TOKEN_FILE_MODE = 0o600

# Hidden prompt shown before a credential read; contains no secret.
_TOKEN_PROMPT = "Enter your ESnet AmSC token (input hidden): "


class CredentialError(Exception):
    """A credential operation failed.  Never carries secret content."""


def amsc_token_path(home):
    """Return the exact persisted token path for ``home``."""
    if not isinstance(home, str) or not home:
        raise CredentialError("a non-empty home directory is required")
    return os.path.join(home, AMSC_TOKEN_RELATIVE_PATH)


def _strip_one_terminal_line_ending(value):
    """Remove only one terminal CRLF/LF; preserve every other byte."""
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith("\n"):
        return value[:-1]
    return value


def _normalize_token(value):
    """Normalize and validate a candidate token without exposing it."""
    if not isinstance(value, str):
        raise CredentialError("token must be a string")
    normalized = _strip_one_terminal_line_ending(value)
    if normalized == "":
        raise CredentialError("token must not be empty")
    return normalized


def _ensure_private_parent(path):
    """Create (or tighten) every directory above ``path`` at mode 0700."""
    parent = os.path.dirname(path)
    current = os.path.abspath(parent)
    missing = []
    while not os.path.isdir(current):
        missing.append(current)
        parent_of_current = os.path.dirname(current)
        if parent_of_current == current:
            break
        current = parent_of_current
    for directory in reversed(missing):
        os.mkdir(directory, mode=_PRIVATE_DIR_MODE)
        os.chmod(directory, _PRIVATE_DIR_MODE)
    os.chmod(parent, _PRIVATE_DIR_MODE)


def _check_candidate_stat(st):
    """Reject unsafe candidate stat results before any open()."""
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise CredentialError(
            "token path must be a regular file (not a symlink or special file)"
        )
    if st.st_uid != os.geteuid():
        raise CredentialError("token file is owned by another user")
    if st.st_mode & 0o077:
        raise CredentialError(
            "token file mode grants group or other access"
        )


def write_amsc_token(home, token):
    """Atomically persist ``token`` at the exact approved path.

    Creates the credential directories at 0700 and the file at 0600.
    Exactly one terminal CRLF/LF is removed before persisting; an empty
    result is rejected.  On a failed replace the previous token stays
    in place, the temp file is removed, and the original OSError
    propagates.  Returns the written path.
    """
    normalized = _normalize_token(token)
    path = amsc_token_path(home)
    _ensure_private_parent(path)
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        os.fchmod(fd, _TOKEN_FILE_MODE)
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(normalized)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    try:
        os.chmod(tmp_path, _TOKEN_FILE_MODE)
        os.replace(tmp_path, path)
    except OSError:
        try:
            if (
                os.path.lexists(tmp_path)
                and os.path.abspath(tmp_path) != os.path.abspath(path)
            ):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return path


def read_amsc_token(home):
    """Return the persisted token after protected lstat checks.

    Rejects symlinks, non-regular files, foreign-owned files, and files
    with group/other access before opening.  Raises CredentialError when
    the file is missing or unsafe.
    """
    path = amsc_token_path(home)
    try:
        st = os.lstat(path)
    except OSError:
        raise CredentialError("AMSC token file is not present")
    _check_candidate_stat(st)
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    return _strip_one_terminal_line_ending(content)


def token_status(home):
    """Return 'absent' | 'present' | 'unsafe' — never token content."""
    path = amsc_token_path(home)
    try:
        st = os.lstat(path)
    except OSError:
        return "absent"
    try:
        _check_candidate_stat(st)
    except CredentialError:
        return "unsafe"
    return "present"


def remove_amsc_token(home):
    """Delete the persisted token.  True if removed, False if absent."""
    path = amsc_token_path(home)
    try:
        os.unlink(path)
    except FileNotFoundError:
        return False
    return True


def prompt_amsc_token(hidden_reader=None, output_stream=None):
    """Read a token via a hidden reader; never echoes the secret.

    ``hidden_reader`` defaults to :func:`getpass.getpass`.  The prompt
    text contains no secret.  Returns the normalized token.
    """
    if hidden_reader is None:
        hidden_reader = getpass.getpass
    if output_stream is not None:
        output_stream.write(_TOKEN_PROMPT + "\n")
    candidate = hidden_reader(_TOKEN_PROMPT)
    return _normalize_token(candidate)


def capture_amsc_token(home, hidden_reader=None, output_stream=None):
    """Prompt once via the hidden reader and persist atomically.

    Returns the written path.  The secret never reaches
    ``output_stream``, the prompt, or any raised message.
    """
    token = prompt_amsc_token(hidden_reader=hidden_reader,
                              output_stream=output_stream)
    return write_amsc_token(home, token)


def reconfigure_amsc_token(home, choice, hidden_reader=None,
                           output_stream=None):
    """Apply an explicit keep/replace/remove choice to a persisted token.

    Returns the resulting ``token_status`` value.  Exposed so a future
    configure flow (and tests of base primitives) can drive
    reconfiguration without the wizard.
    """
    normalized = (choice or "").strip().lower()
    if normalized == "keep":
        return token_status(home)
    if normalized == "remove":
        remove_amsc_token(home)
        return token_status(home)
    if normalized == "replace":
        capture_amsc_token(home, hidden_reader=hidden_reader,
                           output_stream=output_stream)
        return token_status(home)
    raise CredentialError(
        "choice must be one of: keep, replace, remove"
    )
