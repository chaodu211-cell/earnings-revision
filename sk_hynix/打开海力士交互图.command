#!/bin/zsh
cd "${0:A:h}"
if [[ -d "/Applications/Google Chrome.app" ]]; then
  open -a "Google Chrome" "SK海力士_盈利修正.html"
else
  open -a "Safari" "SK海力士_盈利修正.html"
fi
