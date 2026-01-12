# Testing Locally

Install prerequisites, rbenv, and ruby-build:

**On Ubuntu/Debian:**

    sudo apt update
    sudo apt install -y git ruby-full build-essential zlib1g-dev
    sudo apt install rbenv ruby-build

**On Fedora:**

    sudo dnf update
    sudo dnf install -y git ruby ruby-devel @development-tools zlib-devel
    sudo dnf install rbenv ruby-build

Add the following to your .bashrc (and then source ~/.bashrc):

    export PATH="$HOME/.rbenv/bin:$PATH"
    eval "$(rbenv init -)"

Install a version of Ruby via rbenv:

    # list latest stable versions:
    rbenv install -l
    
    # list all local versions:
    rbenv install -L
    
    # install a Ruby version:
    rbenv install 3.1.2

And set the version to use:

    rbenv global 3.1.2   # set the default Ruby version for this machine
    # or:
    rbenv local 3.1.2    # set the Ruby version for this directory

Sanity check:

    ~/projects/mfaulk.github.io (main)$ which ruby
    /home/matt/.rbenv/shims/ruby
    
    ~/projects/mfaulk.github.io (main)$ ruby -v
    ruby 3.1.2p20 (2022-04-12 revision 4491bb740a) [x86_64-linux]
    
    ~/projects/mfaulk.github.io (main)$ gem env home
    /home/matt/.rbenv/versions/3.1.2/lib/ruby/gems/3.1.0
    
With rbenv set up, install Bundler:

    gem install bundler

And serve locally:

    cd yourblog
    bundle install
    bundle exec jekyll serve


