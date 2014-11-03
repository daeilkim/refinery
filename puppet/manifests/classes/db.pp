class db {
  # postgresql-dev required for Python's psycopg2
   /*
  package { [ 'postgresql', 'postgresql-server-dev-all' ]:
    ensure => 'installed',
  }

  service { 'postgresql':
    ensure  => running,
    require => Package[postgresql],
  }

  package {'redis-server':
    ensure => 'installed',
  }
  
  service { 'redis-server':
    ensure  => running,
    require => Package[redis-server],
  }
  */

  class { 'postgresql::server': }

  postgresql::server::db { 'refinery':
    user     => 'vagrant',
    password => postgresql_password('vagrant', ''),
  }

  package { [ 'postgresql-server-dev-all' ]:
    ensure => 'installed',
    require => Class['postgresql::server'],
  }

  # Redis - Server
  package {'redis-server':
    ensure => 'installed',
  }
  
  service { 'redis-server':
    ensure  => running,
    require => Package[redis-server],
  }

}
