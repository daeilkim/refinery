Refinery - A locally deployable open-source web platform for analysis of large document collections open sourced under the MIT License.
===============

* Author - Daeil Kim <daeil.kim@nytimes.com>
* Author - Ben Swanson <dujiaozhu@gmail.com>
[1]: https://github.com/daeilkim/refinery.git

For more information please refer to our help pages in: http://daeilkim.github.io/refinery

What is Refinery?
===============
Built mostly in Python, Refinery is a full-stack web app that works with Vagrant VM and Puppet that can be 
run from a single command after downloading the git repository. 


Installing and Running Refinery
===============
Refinery is a browser driven web application built primarily off of Python. It was developed with the requirement that its implementation process be as simple as possible. Refinery requires three main packages - Git, Virtualbox, and Vagrant VM. VirtualBox and Vagrant VM allows Refinery to exist within a virtual machine that is accessible through your browser. The Vagrant package allows for the deployment of a Puppet manifest, which enables the automated installation of a large number of necessary software modules. 

Before installing, keep in mind that it **requires approximately 2.3GB of hard drive space** and a relatively fast Internet connection. **Installation will run roughly 20-30 minutes of your time using a high-speed Internet connection** so please keep that in mind before installing. Git is needed to clone the repository that will contain the main source code, but if you don't wish to use Git, you can always just download the zip file and uncompress it to a folder you like. However, you'll still need these two pieces of software:

* [Oracle VM VirtualBox (working for v5.0)](https://www.virtualbox.org/)
* [Vagrant (working for v1.8.1)](https://www.vagrantup.com/downloads.html)

To modify the installation process, the configuration file **VagrantFile** located within the root directory contains settings that help guide this process. Installation of Refinery is as follows from the command line:

    git clone https://github.com/daeilkim/refinery.git
    vagrant up

After this command, Refinery will be booting up the virtual machine and loading up the web server. You'll need to then open up any browser and go to this URL: **http://11.11.11.11:8080**. You should see a login screen afterwards which you can login with:

    username: doc
    password: refinery

To see how you Refinery works, you can watch this [video](https://youtu.be/7yRQ1J9Z_LI) which shows a basic run-through using one of the included datasets within the repository.

Licensing
===============
MIT License

Copyright (C) <2014> Daeil Kim

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



