"""The bucket taxonomy. Stage one of routing picks one of these.

Descriptions are what Laya reads as the option criteria, so they are written
as the kinds of request a user makes, not as a list of tool names.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bucket:
    id: str
    description: str
    routable: bool = True  # False: never offered to the user-facing router


BUCKETS: tuple[Bucket, ...] = (
    Bucket(
        "files",
        "browse, list, copy, move, rename, delete or find files and directories; "
        "paths, links, permissions, ownership, disk usage of files",
    ),
    Bucket(
        "text",
        "view, search, filter, count, sort, edit, transform or compare text and "
        "the contents of files; patterns, columns, lines, words",
    ),
    Bucket(
        "archive",
        "compress, decompress, pack or unpack archives such as tar, zip, gzip, bzip2, xz",
    ),
    Bucket(
        "process",
        "running processes and jobs: list, monitor, kill, signals, priorities, "
        "cpu and memory usage, open files, keeping the machine awake, timing a command",
    ),
    Bucket(
        "network",
        "http requests and downloads, remote login and remote copy, ssh keys, "
        "dns lookups, ports, ip addresses, interfaces, wifi, connectivity tests",
    ),
    Bucket(
        "system",
        "operating system and hardware information, date and time, hostname, "
        "power and sleep, kernel settings, system logs, services and daemons, "
        "scheduled tasks, software updates, reboot and shutdown",
    ),
    Bucket(
        "disk",
        "disks, volumes, partitions, mounting and unmounting, disk images, "
        "filesystem checks and formatting, raw device copies, time machine backups",
    ),
    Bucket(
        "users",
        "user accounts and groups, identity, sudo and switching user, passwords, "
        "login sessions, keychain, certificates, encryption and code signing",
    ),
    Bucket(
        "dev",
        "software development: version control such as git, compilers, build "
        "tools, linkers, debuggers, language runtimes and interpreters, package "
        "managers, containers, xcode tooling",
    ),
    Bucket(
        "shell",
        "the shell itself: changing directory, environment variables, aliases, "
        "command history, locating commands, manual pages, loops and conditions, "
        "sleeping, repeating or chaining commands, reading input, printing text",
    ),
    Bucket(
        "macos",
        "the macOS desktop and apps: open files or apps, clipboard, user "
        "preferences and defaults, applescript and automation, spotlight metadata, "
        "quick look, screenshots, text to speech, notifications, shortcuts",
    ),
    Bucket(
        "media",
        "images, audio, video, documents, fonts and printing: convert, resize, "
        "play, inspect metadata, print or manage print queues",
    ),
    Bucket(
        "data",
        "structured data and encodings: sqlite databases, property lists, json, "
        "xml, base64, hex dumps, checksums and hashes, uuids, calculators and unit conversion",
    ),
    Bucket(
        "internal",
        "system daemons, agents and helper binaries that users do not run directly",
        routable=False,
    ),
)

BUCKET_IDS = tuple(b.id for b in BUCKETS)
ROUTABLE = tuple(b for b in BUCKETS if b.routable)


def bucket(bucket_id: str) -> Bucket:
    for b in BUCKETS:
        if b.id == bucket_id:
            return b
    raise KeyError(bucket_id)
