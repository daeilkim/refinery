# Install python and compiled modules for project
class python {
    case $operatingsystem {
        ubuntu: {
            package { "python-pip":
                ensure => installed
            }
            package { ["python-scipy"]:
                ensure => installed,
                require => Package['python-pip']
            }
            package { ["numpy"]:
                ensure => installed,
                provider => pip,
                require => Package['python-pip']
            }


            /*
            package { ['libfreetype6-dev', 'pkg-config']:
                ensure => installed
            }
            package { ['pyparsing']:
                ensure => installed,
                provider => pip,
                require => Package['python-pip']
            }
            package { ["matplotlib"]:
                ensure => installed,
                provider => pip,
                require => Package['numpy', 'pyparsing', 'libfreetype6-dev']
            }
            */


            package { 'virtualenv':
                ensure => installed,
                provider => pip,
                require => Package['python-pip']
            }
            package { 'gunicorn':
                ensure => installed,
                provider => pip,
                require => Package['python-pip']
            }
            package { 'flask':
                ensure => installed,
                provider => pip,
                require => Package['python-pip']
            }
            package { ['joblib','redis','celery']:
                ensure => installed,
                provider => pip,
                require => Package['python-pip']
            }
            package { ['flask-wtf','flask-login','wtforms']:
                ensure => installed,
                provider => pip,
                require => Package['python-pip','flask']
            }
            package { 'scikit-learn':
                provider => pip,
                require => Package['python-pip']
            }
            package { 'kombu':
                ensure => installed,
                provider => pip,
                require => Package['numpy']
            }
            package { ['flask-sqlalchemy','psycopg2']:
                ensure => installed,
                provider => pip,
                require => Class['postgresql::server']
            }
        }
    }
}
