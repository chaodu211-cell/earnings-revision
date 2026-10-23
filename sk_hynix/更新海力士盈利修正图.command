#!/bin/zsh
cd "${0:A:h}"
if python3 build.py --refresh; then
  ./打开海力士交互图.command
else
  print "本次联网更新失败。下面尝试使用原有缓存重建；数据截至日以图上标注为准。"
  if python3 build.py; then
    ./打开海力士交互图.command
  fi
  read "answer?按回车关闭。"
fi
