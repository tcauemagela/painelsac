#!/bin/bash

export GIT_SEQUENCE_EDITOR='sed -i "1s/^pick/reword/; 2s/^pick/reword/"'

git rebase -i --root
