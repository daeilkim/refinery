class run {
  include supervisor
  
  exec { "reset_db":
    command => "/usr/bin/python reset_db.py",
    user => "vagrant",
    cwd    => "/vagrant/refinery/",
  }

  supervisor::app {'celery':
    command => '/usr/local/bin/celery --concurrency=4 -A refinery.celery worker',
    directory => '/vagrant/refinery/',
    user => 'vagrant',
  }

  supervisor::app {'refinery':
    command => '/usr/local/bin/gunicorn --timeout 120 -w 4 -b 0.0.0.0:8080 refinery.webapp.main_menu:app',
    directory => '/vagrant/refinery/',
    user => 'vagrant',
  }

  /*
  exec { "start_celery":
    command => "./start_celery.sh",
    cwd => "/vagrant/refinery/",
    provider => 'shell',
    user => 'vagrant',
    returns => 1,
    require => Exec['reset_db'],
  }

  exec { "start_refinery":
    command => "./start_refinery.sh",
    cwd => "/vagrant/refinery/",
    user => 'vagrant',
    provider => 'shell',
    returns => 1,
    require => Exec['reset_db'],
  }
  */
}