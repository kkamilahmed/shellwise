"""Hand-assigned buckets for the tools people actually type.

Everything listed here is also marked ``common`` in the catalog, which makes
it a candidate in stage two. Tools not listed fall to keyword rules and are
only routable once a plugin registers them.
"""

from __future__ import annotations

_BY_BUCKET: dict[str, str] = {
    "files": """
        ls cp mv rm rmdir mkdir touch find locate mdfind ln readlink realpath stat file
        du df chmod chown chgrp chflags xattr rsync ditto install mkfifo basename dirname
        pwd tree GetFileInfo SetFile fs_usage mktemp truncate
    """,
    "text": """
        cat less more head tail grep egrep fgrep zgrep sed awk sort uniq cut paste join tr
        wc diff cmp comm diff3 sdiff patch column fold fmt nl rev tee expand unexpand look
        split csplit strings iconv vim vi nano ed ex zcat bzcat xzcat col colrm lam pr
    """,
    "archive": """
        tar bsdtar zip unzip zipinfo gzip gunzip gzexe bzip2 bunzip2 bzip2recover xz unxz
        lzma unlzma compress uncompress cpio pax ar
    """,
    "process": """
        ps top kill killall pkill pgrep nice renice nohup lsof time caffeinate sample
        spindump heap leaks vmmap footprint sc_usage purge taskpolicy iostat vm_stat
        memory_pressure jobs bg fg disown wait timeout screen tmux ipcs ipcrm fuser iotop
    """,
    "network": """
        curl ssh scp sftp ssh-keygen ssh-add ssh-agent ssh-copy-id ssh-keyscan ping ping6
        traceroute traceroute6 nc netstat ifconfig arp route dig nslookup host whois
        networksetup scutil ipconfig nettop tcpdump telnet ftp tftp dns-sd networkQuality
        nscurl sntp wget mtr mailx mail ldapsearch snmpwalk rpcinfo lwp-request
    """,
    "system": """
        uname sw_vers system_profiler sysctl hostname uptime date cal ncal pmset systemsetup
        softwareupdate launchctl log syslog dmesg reboot shutdown halt periodic crontab at
        atq atrm batch ioreg kextstat kmutil nvram locale powermetrics spctl profiles
        csrutil
    """,
    "disk": """
        diskutil mount umount hdiutil dd fsck fsck_apfs fsck_hfs fsck_msdos newfs_apfs
        newfs_hfs newfs_msdos tmutil bless asr fdisk gpt
    """,
    "users": """
        whoami id groups users who w last passwd su sudo visudo dscl dseditgroup dscacheutil
        login chpass chfn chsh security codesign openssl certtool mkpassd chroot finger htpasswd
    """,
    "dev": """
        git make cmake gcc cc g++ clang clang++ c++ ld lldb python3 python perl ruby irb gem
        bundle bundler swift swiftc xcodebuild xcode-select xcrun java javac pip3 tclsh m4
        yacc bison flex lex nm otool strip lipo dsymutil install_name_tool libtool ranlib
        size ctags dtrace dtruss agvtool actool ibtool docker gh node npm brew go cargo
        rustc gcov jps rails
    """,
    "shell": """
        cd echo printf export unset set unalias alias source eval exec exit return history
        fc type which whereis whatis apropos man command builtin hash read test true false
        sleep seq yes xargs env tty clear reset script expr let shift getopt getopts trap
        bindkey typeset declare local readonly emulate setopt unsetopt dirs pushd popd
        umask ulimit limit unlimited watch sh bash dash zsh [ printenv tput stty
    """,
    "macos": """
        open defaults osascript osacompile osalang pbcopy pbpaste say screencapture qlmanage
        mdls mdimport mdutil shortcuts automator tccutil notifyutil
    """,
    "media": """
        sips afplay afconvert afinfo afclip textutil cupsfilter lp lpr lpq lprm lpstat cancel
        lpadmin lpoptions lpinfo cupsctl cupsenable cupsdisable cupsaccept cupsreject
        avmetareadwrite auval auvaltool enscript ffmpeg convert
    """,
    "data": """
        sqlite3 plutil xmllint xsltproc base64 md5 md5sum shasum sha1sum sha224sum sha256sum
        sha384sum sha512sum cksum sum uuidgen hexdump od xxd bc dc units uuencode uudecode
        jq
    """,
}

CURATED: dict[str, str] = {
    tool: bucket for bucket, names in _BY_BUCKET.items() for tool in names.split()
}
