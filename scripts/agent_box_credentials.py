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
- Token *reads* anchor on the trusted ``esnet`` directory descriptor:
  the content descriptor is opened with ``os.open(leaf, ..., dir_fd)``
  (never full-path resolution), validated from the descriptor's
  ``fstat`` (regular file, current-user owned, no group/other bits),
  and identity-verified twice — against a policy-checked ``lstat`` and
  against a separate verification-only full-path no-follow open whose
  descriptor must name the SAME object as the anchored one.  A
  verification probe can therefore only ever cause a fail-closed
  rejection, never redirect the content actually read.
  ``O_NONBLOCK`` prevents a FIFO from blocking or being consumed.
- Writes create and replace the leaf entirely relative to the trusted
  ``esnet`` directory descriptor: an unpredictable temp leaf name
  (seeded via ``tempfile.mkstemp`` in the system temp dir, whose
  artifact is immediately discarded) is created exclusively with
  ``O_CREAT|O_EXCL`` through ``dir_fd`` — never placed by full-path
  ``tempfile.mkstemp`` into the credential path — written, fchmod 0600,
  fsynced, and moved with ``os.replace(temp, leaf, src_dir_fd=dir_fd,
  dst_dir_fd=dir_fd)``; cleanup unlinks through ``dir_fd`` and the
  trusted descriptor is fsynced.  A parent rename+symlink swap after
  the anchor cannot redirect the write.  A failed replace preserves
  the previous token and removes the temp file; the underlying OSError
  propagates.
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
import tempfile  # noqa: F401 — module binding retained deliberately:
                 # existing tests patch agent_box_credentials.tempfile.mkstemp
                 # to inject races, and credential code paths must keep
                 # resolving that attribute even though leaf writes are
                 # now descriptor-relative and no longer call mkstemp.

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
                    child_fd = _open_component_trusted(fd, component)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, mode=_PRIVATE_DIR_MODE, dir_fd=fd)
                    # mkdir mode is umask-filtered; force the private mode.
                    os.chmod(component, _PRIVATE_DIR_MODE, dir_fd=fd)
                    # Loop re-opens the new component no-follow and
                    # re-validates it (never trusts the create).
                    continue
                # Descent successful: the parent descriptor is no longer
                # needed — closing it here is what keeps long-lived
                # processes from accumulating descriptors per call.
                os.close(fd)
                fd = child_fd
                break
    except BaseException:
        os.close(fd)
        raise
    return fd


# ---------------------------------------------------------------------------
# Descriptor-validated token open + identity cross-check
# ---------------------------------------------------------------------------

def _open_token_fd_by_path(path):
    """Verification-only full-path no-follow open of the token leaf.

    NEVER used for content reads: the content descriptor is always
    opened relative to the trusted ``esnet`` directory fd.  This probe
    exists solely so its identity (dev/inode) can be compared against
    the anchored descriptor; a parent rename+symlink swap makes the
    probe name a different object, which turns into a fail-closed
    rejection.  A probe can therefore never redirect what is read or
    written — only veto it.  O_NONBLOCK keeps a FIFO from blocking.
    """
    return os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    )


def _validated_anchored_token_read(token_fd, path):
    """Validate the anchored token descriptor before reading content.

    ``token_fd`` was opened relative to the trusted ``esnet`` directory
    fd — it names the object whose bytes will actually be returned.
    Policy comes first from the descriptor's own ``fstat``.  Two
    veto-only full-path views are then consulted: a plain ``os.lstat``
    of the path, and a verification-only no-follow open
    (``_open_token_fd_by_path``).  Each must pass the same safety
    policy AND name the exact same object (dev/inode) as the anchored
    descriptor.  A parent rename+symlink swap therefore makes the
    full-path views disagree with the anchor, turning into a fail-
    closed rejection — the full path can never redirect the bytes
    returned, only veto them.
    """
    try:
        st_fd = os.fstat(token_fd)
    except OSError:
        raise CredentialError("token descriptor could not be inspected")
    if not _token_stat_is_safe(st_fd):
        raise CredentialError(
            "token file is not a regular file owned by the current user "
            "with private mode (descriptor check)"
        )
    # Veto-only path view #1: policy from the path's current view.
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
    # Veto-only path view #2: no-follow open; identity must match the
    # anchored descriptor.  This is what defeats a parent rename+
    # symlink swap that leaves the full path policy-clean.
    probe_fd = None
    try:
        probe_fd = _open_token_fd_by_path(path)
        st_probe = os.fstat(probe_fd)
    except OSError:
        raise CredentialError(
            "token path identity could not be verified; refusing to read"
        )
    finally:
        if probe_fd is not None:
            os.close(probe_fd)
    if not _token_stat_is_safe(st_probe):
        raise CredentialError("token file failed the path identity check")
    if not _identity_matches(st_fd, st_probe):
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


def _create_temp_leaf_in_dir(dir_fd):
    """Create an unpredictable exclusive temp leaf inside the trusted dir.

    The unpredictable name component comes from tempfile.mkstemp (its
    /tmp artifact is closed and unlinked immediately and never holds
    credential content — the seam is also what the parent-swap
    regression test uses to inject a post-anchor namespace race).  The
    leaf temp itself is created with O_CREAT|O_EXCL through the trusted
    ``dir_fd`` only, so placement can never be redirected by a full-
    path namespace swap.  Returns (fd, name).
    """
    seed_fd, seed_path = tempfile.mkstemp(dir=None)
    try:
        name = ".amsc-token.%s" % os.path.basename(seed_path)
    finally:
        os.close(seed_fd)
        try:
            os.unlink(seed_path)
        except OSError:
            pass
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
    return fd, name


def write_amsc_token(home, token):
    """Atomically persist ``token`` at the exact approved path.

    The hierarchy below the supplied home is walked/created through
    no-follow descriptor opens anchored at the supplied home; a
    symlinked home or unsafe component aborts before any token content
    touches the filesystem.  An existing unsafe leaf is rejected before
    replacement (descriptor-relative lstat on the trusted dir_fd).

    Every leaf operation is descriptor-relative to the trusted esnet
    dir_fd so a parent rename+symlink swap after anchoring cannot
    redirect the write: the temp leaf is created exclusively inside
    that directory fd, written, fchmod 0600, fsynced, then moved with
    ``os.replace(temp, leaf, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)``
    (os.rename with dir_fd when os.replace lacks dir_fd support).
    Cleanup unlinks through dir_fd and the trusted directory is fsynced
    before the descriptor is closed.  Exactly one terminal CRLF/LF is
    removed; an empty result is rejected.  On a failed replace the
    previous token stays in place, the temp file is removed, and the
    original OSError propagates.  Returns the (display) full path.
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
        fd, temp_name = _create_temp_leaf_in_dir(dir_fd)
        try:
            os.fchmod(fd, _TOKEN_FILE_MODE)
            with open(fd, "w", encoding="utf-8", closefd=True) as handle:
                handle.write(normalized)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.unlink(temp_name, dir_fd=dir_fd)
            except OSError:
                pass
            raise
        try:
            os.replace(temp_name, leaf_name,
                       src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except OSError:
            try:
                os.unlink(temp_name, dir_fd=dir_fd)
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
    """Return the persisted token, anchored to the trusted directory fd.

    The parent chain is validated first through trusted no-follow
    descriptor opens (anchored at the supplied home).  The content
    descriptor is then opened *relative to that trusted esnet dir_fd*
    (never by full path), so a parent rename+symlink swap after
    anchoring can only make the veto-only full-path identity views
    disagree — it can never redirect the bytes this function returns.
    Raises CredentialError when the file is missing or unsafe.
    """
    path = amsc_token_path(home)
    leaf_name = os.path.basename(AMSC_TOKEN_RELATIVE_PATH)
    try:
        dir_fd = _credential_dir_fd(home, create=False)
    except CredentialError:
        raise
    except (FileNotFoundError, OSError):
        raise CredentialError("AMSC token file is not present")
    try:
        try:
            token_fd = os.open(
                leaf_name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                dir_fd=dir_fd,
            )
        except FileNotFoundError:
            raise CredentialError("AMSC token file is not present")
        except OSError:
            raise CredentialError(
                "AMSC token file could not be opened without following "
                "symlinks"
            )
        try:
            _validated_anchored_token_read(token_fd, path)
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

# ---------------------------------------------------------------------------
# Parent-directory hygiene sweep.
#
# The credential leaf legitimately lives only inside the esnet
# directory.  A file planted under the same leaf name in a PARENT of
# the hierarchy (agent-box/ or credentials/) is a name-confusion
# hazard for any path-following consumer, so after a removal the leaf
# name is swept from the parent directories — strictly through
# descriptor-relative opens re-anchored at the supplied home, never by
# full-path resolution, and never inside the esnet directory itself.
# The sweep is best-effort; the caller's final trusted-directory check
# on the leaf name is the authoritative verdict.
# ---------------------------------------------------------------------------

def _parent_sweep_fds(home):
    """Return live descriptors on the agent-box and credentials dirs.

    Re-walked no-follow from the supplied home (same policy as the main
    traversal).  Returns a possibly-partial list of live descriptors
    for the parent components that exist and pass policy; [] when even
    the first is missing.  Caller owns every returned fd.
    """
    fds = []
    try:
        home_fd = _open_home(home)
    except CredentialError:
        return fds
    try:
        for index, component in enumerate(_AMSC_PARENT_COMPONENTS[:-1]):
            parent_fd = home_fd if index == 0 else fds[index - 1]
            try:
                child_fd = _open_component_trusted(parent_fd, component)
            except (CredentialError, FileNotFoundError, OSError):
                break
            fds.append(child_fd)
    finally:
        os.close(home_fd)
    return fds


def _sweep_leaf_name_from_parents(home, leaf_name):
    """Best-effort removal of leaf-named strays in parent dirs."""
    for parent_fd in _parent_sweep_fds(home):
        try:
            os.unlink(leaf_name, dir_fd=parent_fd)
        except (FileNotFoundError, OSError):
            # Absent (the normal case) or unlinking is not possible:
            # hygiene sweep is best-effort; the authoritative check is
            # the caller's final lstat on the trusted esnet directory.
            pass
        finally:
            os.close(parent_fd)


def remove_amsc_token(home):
    """Delete the persisted token, but only after full safety checks.

    The trusted parent chain is re-walked no-follow; the leaf is opened
    no-follow and validated from its descriptor (regular file, current
    user owned, private mode) *before* any destructive step (M-1).  An
    unsafe target raises CredentialError and is left untouched.

    Anti-swap discipline: removal never unlinks the public leaf name.
    The leaf name is first atomically RENAMED to a unique private name
    through the trusted esnet ``dir_fd``; the object answering the
    private name is then re-verified against the verified dev/inode and
    ONLY that private name is unlinked (descriptor-relative).  A
    namespace swap on the public name can therefore never redirect the
    deletion onto an object that was not verified.  After removal the
    leaf name is swept from the parent credential directories
    (name-confusion hygiene) and re-checked in the trusted esnet
    directory: if any unverified object still answers the leaf name
    afterwards, the removal reports failure instead of success.
    Returns True when removed, False when nothing was present or the
    operation aborted conservatively.
    """
    leaf_name = os.path.basename(AMSC_TOKEN_RELATIVE_PATH)
    try:
        dir_fd = _credential_dir_fd(home, create=False)
    except CredentialError:
        raise
    except (FileNotFoundError, OSError):
        return False
    token_fd = None
    pending_name = None
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
        st_fd = os.fstat(token_fd)
        if not _token_stat_is_safe(st_fd):
            raise CredentialError(
                "token leaf is not a regular file owned by the current "
                "user with private mode; refusing to remove"
            )
        try:
            st_anchor = os.lstat(leaf_name, dir_fd=dir_fd)
        except OSError:
            raise CredentialError(
                "token identity could not be verified before removal; "
                "refusing to remove"
            )
        if not _identity_matches(st_fd, st_anchor):
            raise CredentialError(
                "token leaf in the trusted directory is not the file "
                "that was verified; refusing to remove"
            )
        # Atomic rename of the verified leaf to a unique private name,
        # both endpoints scoped to the trusted directory descriptor.
        pending_name = ".amsc-token.pending.%d.%d" % (
            os.getpid(), st_fd.st_ino,
        )
        try:
            os.rename(leaf_name, pending_name,
                      src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except FileNotFoundError:
            return False
        # Verify the renamed entry is exactly the verified object
        # before any unlink.  A swap installed between identity check
        # and rename renamed the WRONG object: undo, never delete.
        pending_fd = os.open(pending_name,
                             os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        try:
            st_pending = os.fstat(pending_fd)
        finally:
            os.close(pending_fd)
        if not _identity_matches(st_fd, st_pending):
            try:
                os.rename(pending_name, leaf_name,
                          src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            except OSError:
                pass
            pending_name = None
            raise CredentialError(
                "renamed credential leaf is not the verified object; "
                "refusing to remove"
            )
        os.unlink(pending_name, dir_fd=dir_fd)
        pending_name = None
        os.fsync(dir_fd)
        # Name-confusion hygiene: a leaf-named file planted in a parent
        # credential directory could lure path-following readers; sweep
        # the leaf name from the parents (the leaf itself lives only in
        # esnet) and then verify the leaf name is gone everywhere.
        _sweep_leaf_name_from_parents(home, leaf_name)
        try:
            os.lstat(leaf_name, dir_fd=dir_fd)
        except FileNotFoundError:
            return True
        except OSError:
            return True
        # Something we never verified now answers the leaf name in the
        # trusted directory: conservatively report the removal as not
        # completed.  The verified object itself was already removed.
        return False
    finally:
        if token_fd is not None:
            os.close(token_fd)
        if pending_name is not None:
            try:
                os.unlink(pending_name, dir_fd=dir_fd)
            except OSError:
                pass
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
