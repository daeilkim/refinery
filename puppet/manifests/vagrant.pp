# This vagrant.pp represents the base script to begin installation of Refinery

import "classes/*.pp"

$PROJ_DIR = "/vagrant"
$HOME_DIR = "/home/vagrant"

Exec {
    path => "/usr/local/bin:/usr/bin:/usr/sbin:/sbin:/bin",
}

class dev {
    class {
        init: ;
        db: require => Class[init];
        python: require => Class["init","db"];
        run: require => Class["init", "db", "python"];
    }
}

include dev
