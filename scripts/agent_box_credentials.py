"""Secure capture and storage of the ESnet AmSC token (Task 3 + security).

Single persisted location, relative to a supplied home:
``agent-box/credentials/esnet/amsc-token`` with every credential
directory held at mode 0700 and the token file at mode 0600.

Trust-boundary rules enforced here (Aegis I-1 / I-3 / M-1):

- The credential hierarchy is anchored at the *supplied* home.  The
  home itself must be a real directory (O_NOFOLLOW open — a symlinked
  home is rejected).  Each of ``agent-box``, ``credentials`` and
  ``esnet`` must be a real directory (never a symlink, never a
  non-directory), owned by the current user, and carry no group/other
  permission bits.  Components are validated and created
  component-by-component through descriptor-relative O_NOFOLLOW opens
  (no lstat-then-act on the full path).
- Token *reads* open the file no-follow and validate from the
  ``fstat`` of the resulting descriptor (regular file, current-user
  owned, no group/other bits) — not from a pre-open lstat of the path —
  then cross-check the descriptor against a fresh ``lstat`` of the path
  (identity: same dev/inode, same safety policy result) so a path swap
  between validation and use cannot silently read an attacker file.
  ``O_NONBLOCK`` prevents a FIFO at the path from blocking or being
  consumed.
- Writes walk/create the hierarchy through trusted directory
  descriptors (missing components created 0700 via ``dir_fd`` and
  re-opened no-follow for verification), reject an existing unsafe leaf
  *before* replacement, then perform the approved same-directory atomic
  sequence: mkstemp + write + fsync + 0600 + ``os.replace`` + directory
  fsync.  A failed replace preserves the previous token and removes the
  temp file; the underlying OSError propagates.
- ``remove_amsc_token`` (M-1) re-validates the trusted parent chain and
  the leaf (no-follow open + ``fstat``: regular, owned, private)
  *before* unlinking through the trusted directory descriptor; an
  unsafe target is rejected, never removed.
- Exactly one terminal line ending (CRLF or LF) is removed from a
  captured value; all other content, including leading/trailing
  whitespace, is preserved verbatim.  Empty values are rejected.
- Interactive capture uses a hidden reader (``getpass.getpass`` by
  default) and never writes the secret to any output stream, prompt,
  exception, log, argument, or environment variable.
- ``token_status`` reports only ``"absent"``, ``"present"`` or
  ``"unsafe"`` — it never returns or displays token content.
- External operator-supplied token files are read through
  ``read_external_token_file``, which applies the same descriptor-based
  no-follow policy (parent opened no-follow and required to be a real,
  current-user-owned directory; file opened no-follow and validated
  from its descriptor).

Portability: every operation requires ``os.O_NOFOLLOW`` and
descriptor-relative ``dir_fd`` support and fails closed with
CredentialError when the platform lacks them; the trust boundary is
never silently weakened.

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

# Components of the credential hierarchy below the supplied home, in
# walk/creation order; the leaf token file is not included.
_AMSC_PARENT_COMPONENTS = ("agent-box", "credentials", "esnet")

# Private mode for every directory in the credential chain.
_PRIVATE_DIR_MODE = 0o700

# Private mode for the token file itself.
_TOKEN_FILE_MODE = 0o600

# Hidden prompt shown before a credential read; contains no secret.
_TOKEN_PROMPT = "Enter your ESnet AmSC token (input hidden): "


class CredentialError(Exception):
    """A credential operation failed.  Never carries secret content."""


def _require_nofollow():
    """Fail closed unless no-follow descriptor-relative operations exist.

    Portability contract (Aegis note): every credential operation
    requires ``os.O_NOFOLLOW``, ``os.O_DIRECTORY`` and ``dir_fd``
    support on ``os.open``/``os.mkdir``/``os.unlink``/``os.lstat``.  On
    a platform lacking any of them the credential store refuses to
    operate instead of silently falling back to path-based (symlink-
    following) equivalents.
    """
    if not (
        getattr(os, "O_NOFOLLOW", 0)
        and getattr(os, "O_DIRECTORY", 0)
        and callable(getattr(os, "open", None))
        and callable(getattr(os, "mkdir", None))
        and callable(getattr(os, "unlink", None))
        and callable(getattr(os, "lstat", None))
    ):
        raise CredentialError(
            "platform lacks O_NOFOLLOW/dir_fd support; refusing to "
            "operate with a weakened trust boundary"
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Descriptor-based safety predicates (policy applied to the SAME stat
# result the decision is made from — no lstat/open TOCTOU)
# ---------------------------------------------------------------------------

def _dir_stat_is_safe(st):
    """A trusted credential directory: real dir, owned, private."""
    return (
        stat.S_ISDIR(st.st_mode)
        and st.st_uid == os.geteuid()
        and not (st.st_mode & 0o077)
    )


def _home_stat_is_valid(st):
    """Supplied home must be a real directory (mode/owner not imposed:
    real user homes are commonly 0755; the credential components carry
    the private-mode requirement)."""
    return stat.S_ISDIR(st.st_mode)


def _token_stat_is_safe(st):
    """A trusted token object: regular file, owned, no group/other bits."""
    return (
        stat.S_ISREG(st.st_mode)
        and st.st_uid == os.geteuid()
        and not (st.st_mode & 0o077)
    )


def _identity_matches(st_a, st_b):
    return st_a.st_dev == st_b.st_dev and st_a.st_ino == st_b.st_ino


# ---------------------------------------------------------------------------
# Trusted hierarchy traversal (component-by-component, O_NOFOLLOW+dir_fd)
# ---------------------------------------------------------------------------

def _open_home(home):
    """Open the supplied home no-follow; reject a symlinked home."""
    if not isinstance(home, str) or not home:
        raise CredentialError("a non-empty home directory is required")
    root = home if os.path.isabs(home) else os.path.abspath(home)
    try:
        fd = os.open(root, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        # ELOOP (symlink home), ENOENT, EACCES — reject without following.
        raise CredentialError(
            "credential home could not be opened without following "
            "symlinks"
        )
    try:
        st = os.fstat(fd)
        if not _home_stat_is_valid(st):
            raise CredentialError(
                "credential home is not a real directory (symlinks are "
                "rejected)"
            )
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_component_trusted(parent_fd, component):
    """Open one credential component relative to a trusted parent fd.

    Raises FileNotFoundError when missing; CredentialError when present
    but unsafe (symlink, non-directory, foreign owner, or any
    group/other bits).
    """
    try:
        fd = os.open(
            component,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        raise
    except NotADirectoryError:
        raise CredentialError(
            "credential component %r exists but is not a directory"
            % (component,)
        )
    except OSError:
        # ELOOP (symlink), EACCES, etc. — reject without following.
        raise CredentialError(
            "credential component %r could not be opened without "
            "following symlinks" % (component,)
        )
    try:
        st = os.fstat(fd)
        if not _dir_stat_is_safe(st):
            raise CredentialError(
                "credential component %r is owned by another user or "
                "grants group/other access" % (component,)
            )
    except BaseException:
        os.close(fd)
        raise
    return fd


def _credential_dir_fd(home, create):
    """Descriptor on ``<home>/agent-box/credentials/esnet`` — trusted.

    Walks the hierarchy component-by-component anchored at the supplied
    home.  With ``create`` missing components are made 0700 through the
    trusted parent descriptor and re-opened no-follow for verification;
    without ``create`` a missing component raises FileNotFoundError.
    Unsafe components raise CredentialError.  Caller owns the fd.
    """
    _require_nofollow()
    fd = _open_home(home)
    try:
        for component in _AMSC_PARENT_COMPONENTS:
            while True:
                try:
                    fd = _open_component_trusted(fd, component)
                    break
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, mode=_PRIVATE_DIR_MODE, dir_fd=fd)
                    # mkdir mode is umask-filtered; force the private mode.
                    os.chmod(component, _PRIVATE_DIR_MODE, dir_fd=fd)
                    # Loop re-opens the new component no-follow and
                    # re-validates it (never trusts the create).
    except BaseException:
        os.close(fd)
        raise
    return fd


# ---------------------------------------------------------------------------
# Descriptor-validated token open + identity cross-check
# ---------------------------------------------------------------------------

def _open_token_fd_by_path(path):
    """Open the token by full path, no-follow last component, non-blocking.

    The caller must first have validated the parent chain via
    ``_credential_dir_fd`` so path resolution cannot escape through a
    symlinked component.  O_NONBLOCK keeps a FIFO at the path from
    blocking; the descriptor fstat then rejects it.
    """
    return os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    )


def _validated_token_fd_read(token_fd, path):
    """Validate descriptor + path identity, then read the token."""
    try:
        st_fd = os.fstat(token_fd)
    except OSError:
        raise CredentialError("token descriptor could not be inspected")
    if not _token_stat_is_safe(st_fd):
        raise CredentialError(
            "token file is not a regular file owned by the current user "
            "with private mode (descriptor check)"
        )
    # Identity cross-check: the object the path still names must be the
    # exact object this descriptor refers to, and the path view must
    # pass the same policy.  A swap between open and use is rejected.
    try:
        st_path = os.lstat(path)
    except OSError:
        raise CredentialError(
            "token path identity could not be verified; refusing to read"
        )
    if not _token_stat_is_safe(st_path):
        raise CredentialError(
            "token file failed the path identity safety check"
        )
    if not _identity_matches(st_fd, st_path):
        raise CredentialError(
            "token file on disk is not the file that was opened; "
            "refusing to read"
        )
    return st_fd


def _read_fd_content(token_fd):
    try:
        with open(token_fd, "r", encoding="utf-8", closefd=True) as handle:
            content = handle.read()
    except OSError:
        raise CredentialError("token file could not be read")
    return _strip_one_terminal_line_ending(content)


# ---------------------------------------------------------------------------
# Public API — path / status
# ---------------------------------------------------------------------------

def amsc_token_path(home):
    """Return the exact persisted token path for ``home``."""
    if not isinstance(home, str) or not home:
        raise CredentialError("a non-empty home directory is required")
    return os.path.join(home, AMSC_TOKEN_RELATIVE_PATH)


def token_status(home):
    """Return 'absent' | 'present' | 'unsafe' — never token content."""
    try:
        _require_nofollow()
    except CredentialError:
        return "unsafe"
    try:
        dir_fd = _credential_dir_fd(home, create=False)
    except CredentialError:
        return "unsafe"
    except (FileNotFoundError, OSError):
        return "absent"
    try:
        try:
            token_fd = os.open(
                os.path.basename(AMSC_TOKEN_RELATIVE_PATH),
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                dir_fd=dir_fd,
            )
        except FileNotFoundError:
            return "absent"
        except OSError:
            return "unsafe"
        try:
            st = os.fstat(token_fd)
        finally:
            os.close(token_fd)
        return "present" if _token_stat_is_safe(st) else "unsafe"
    finally:
        os.close(dir_fd)


# ---------------------------------------------------------------------------
# Public API — write (atomic, hierarchy-trusted, unsafe-leaf rejection)
# ---------------------------------------------------------------------------

def _reject_existing_unsafe_leaf(dir_fd, leaf_name):
    """Reject an existing leaf before replacement (I-1), via the trusted
    directory descriptor so the check cannot be fooled by path swaps."""
    try:
        st = os.lstat(leaf_name, dir_fd=dir_fd)
    except FileNotFoundError:
        return
    except OSError:
        raise CredentialError(
            "existing token could not be inspected; refusing to overwrite"
        )
    if stat.S_ISLNK(st.st_mode):
        raise CredentialError(
            "existing token is a symlink; refusing to overwrite"
        )
    if not _token_stat_is_safe(st):
        raise CredentialError(
            "existing token is not a regular file owned by the current "
            "user with private mode; refusing to overwrite"
        )


def write_amsc_token(home, token):
    """Atomically persist ``token`` at the exact approved path.

    The hierarchy below the supplied home is walked/created through
    no-follow descriptor opens anchored at the supplied home; a
    symlinked home or unsafe component aborts before any token content
    touches the filesystem.  An existing unsafe leaf is rejected before
    replacement.  The approved atomic sequence (same-directory mkstemp,
    fsync, 0600, ``os.replace``, directory fsync) then persists the
    token.  Exactly one terminal CRLF/LF is removed; an empty result is
    rejected.  On a failed replace the previous token stays in place,
    the temp file is removed, and the original OSError propagates.
    Returns the written path.
    """
    normalized = _normalize_token(token)
    path = amsc_token_path(home)
    leaf_name = os.path.basename(AMSC_TOKEN_RELATIVE_PATH)
    try:
        dir_fd = _credential_dir_fd(home, create=True)
    except FileNotFoundError:
        raise CredentialError(
            "credential home does not exist; refusing to create it"
        )
    try:
        _reject_existing_unsafe_leaf(dir_fd, leaf_name)
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
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return path


# ---------------------------------------------------------------------------
# Public API — protected reads
# ---------------------------------------------------------------------------

def read_amsc_token(home):
    """Return the persisted token after descriptor-validated no-follow
    open and descriptor/path identity cross-check (I-1).

    The parent chain is validated first through trusted no-follow
    descriptors (anchored at the supplied home), so path resolution
    cannot escape via a symlinked home or component.  The file itself
    is then opened no-follow and validated from the descriptor's
    fstat, and cross-checked against the path via lstat identity.
    Raises CredentialError when the file is missing or unsafe.
    """
    path = amsc_token_path(home)
    try:
        dir_fd = _credential_dir_fd(home, create=False)
    except CredentialError:
        raise
    except (FileNotFoundError, OSError):
        raise CredentialError("AMSC token file is not present")
    try:
        try:
            token_fd = _open_token_fd_by_path(path)
        except FileNotFoundError:
            raise CredentialError("AMSC token file is not present")
        except OSError:
            raise CredentialError(
                "AMSC token file could not be opened without following "
                "symlinks"
            )
        try:
            _validated_token_fd_read(token_fd, path)
        except BaseException:
            os.close(token_fd)
            raise
        return _read_fd_content(token_fd)
    finally:
        os.close(dir_fd)


def read_external_token_file(path):
    """Securely read an operator-supplied external token file (I-1).

    Applies the same descriptor-based, no-follow policy as the
    persisted path: the parent directory is opened O_NOFOLLOW and must
    be a real, current-user-owned directory; the file is opened
    no-follow and validated from the descriptor's fstat (regular,
    owned, no group/other bits).  Returns the normalized token content;
    never echoes it anywhere.
    """
    _require_nofollow()
    if not isinstance(path, str) or not path or not os.path.isabs(path):
        raise CredentialError(
            "external token path must be a non-empty absolute path"
        )
    parent = os.path.dirname(path)
    leaf = os.path.basename(path)
    if not leaf:
        raise CredentialError("external token path has no file component")
    try:
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        raise CredentialError(
            "external token parent could not be opened as a real "
            "directory (symlinks are rejected)"
        )
    try:
        try:
            parent_st = os.fstat(parent_fd)
        except OSError:
            raise CredentialError(
                "external token parent could not be inspected"
            )
        if not (
            stat.S_ISDIR(parent_st.st_mode)
            and parent_st.st_uid == os.geteuid()
        ):
            raise CredentialError(
                "external token parent is not a real directory owned by "
                "the current user"
            )
        try:
            token_fd = os.open(
                leaf,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
        except OSError:
            raise CredentialError(
                "external token file could not be opened without "
                "following symlinks"
            )
        try:
            st = os.fstat(token_fd)
            if not _token_stat_is_safe(st):
                raise CredentialError(
                    "external token file is not a regular file owned by "
                    "the current user with private mode"
                )
        except BaseException:
            os.close(token_fd)
            raise
        return _read_fd_content(token_fd)
    finally:
        os.close(parent_fd)


# ---------------------------------------------------------------------------
# Public API — explicit remove (M-1: validated before unlink)
# ---------------------------------------------------------------------------

def remove_amsc_token(home):
    """Delete the persisted token, but only after full safety checks.

    The trusted parent chain is re-walked no-follow, and the leaf is
    opened no-follow and validated from its descriptor (regular file,
    current-user owned, private mode) *before* unlinking through the
    trusted directory descriptor.  An unsafe target is rejected with
    CredentialError and left in place.  Returns True when removed,
    False when nothing was present.
    """
    leaf_name = os.path.basename(AMSC_TOKEN_RELATIVE_PATH)
    try:
        dir_fd = _credential_dir_fd(home, create=False)
    except CredentialError:
        raise
    except (FileNotFoundError, OSError):
        return False
    try:
        try:
            token_fd = os.open(
                leaf_name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                dir_fd=dir_fd,
            )
        except FileNotFoundError:
            return False
        except OSError:
            raise CredentialError(
                "token leaf could not be opened without following "
                "symlinks; refusing to remove"
            )
        try:
            st = os.fstat(token_fd)
            if not _token_stat_is_safe(st):
                raise CredentialError(
                    "token leaf is not a regular file owned by the "
                    "current user with private mode; refusing to remove"
                )
        finally:
            os.close(token_fd)
        try:
            os.unlink(leaf_name, dir_fd=dir_fd)
        except FileNotFoundError:
            return False
        os.fsync(dir_fd)
        return True
    finally:
        os.close(dir_fd)


# ---------------------------------------------------------------------------
# Public API — interactive capture
# ---------------------------------------------------------------------------

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
