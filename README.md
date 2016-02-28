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

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



