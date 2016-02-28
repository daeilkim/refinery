Refinery - A locally deployable open-source web platform for analysis of large document collections
===============

* Author - Daeil Kim <daeil.kim@nytimes.com>
* Author - Ben Swanson <dujiaozhu@gmail.com>
[1]: https://github.com/daeilkim/refinery.git

What is Refinery?
===============
Built mostly in Python, Refinery is a full-stack web app that works with Vagrant VM and Puppet that can be 
run from a single command after downloading the git repository. The web-app is accessed through 


Pre-Requisites
===============
You'll need both the Oracle VM VirtualBox software and Vagrant installed which you can find here as well as git.  

* [Git][1]
* [Oracle VM VirtualBox (working for v5.0)][2]
* [Vagrant (working for v1.8.1)][3]

[1]: http://git-scm.com/
[2]: https://www.virtualbox.org/ 
[3]: http://vagrantup.com/ 


Running Refinery
===============
Once you have both the virtual box and vagrant installed, simply git clone the refinery repository

1. Git clone this repository to download necessary vagrant files

    ```
    git clone https://github.com/daeilkim/refinery.git
    ```

2. Start the Refinery Vagrant VM from the root folder of refinery. This will take a while as it is downloading all the necessary packages as well as configuring these dependencies and running the web-app as a service.

    ```
    vagrant up
    ```

3. Open up any browser and go to this address (username/password: doc/refinery): 

    ```
    http://11.11.11.11:8080
    ```
    
Copyright (C) <2014>  <Daeil Kim>
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA




