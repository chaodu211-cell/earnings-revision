#!/bin/bash
# 双击我：同步 GitHub 上的记录 → 抓新的 FactSet 周报和纳指行情 → 重算超额修正 / 速度拐点 → 打开「修正速度与纳指.html」
#
# 网络走 Veee 的系统代理（自动读端口）。没开代理或数据源暂时连不上时，用本地已有数据照样重算并打开，
# 页面顶上会写明数据截至哪天。

cd "$(dirname "$0")" || { echo "无法进入脚本所在目录"; exit 1; }

die() {
  echo
  echo "────────────────────────────────────────"
  echo "❌ $*"
  echo "────────────────────────────────────────"
  echo
  read -n 1 -s -r -p "按任意键关闭此窗口…"
  echo
  exit 1
}

echo "════════════════════════════════════════"
echo "  盈利修正 · 修正速度与纳指 · 更新并打开"
echo "  目录：$(pwd)"
echo "════════════════════════════════════════"
echo

PY="$(command -v python3 2>/dev/null)"
[ -n "$PY" ] || die "找不到 python3（xcode-select --install 后重试）"
"$PY" -c 'import pandas, numpy, scipy, fitz, yfinance' 2>/dev/null \
  || die "缺依赖。在终端里跑：$PY -m pip install --user pandas numpy scipy pymupdf yfinance"

# ---------- 代理：读系统代理（Veee）----------
if [ "$(scutil --proxy | awk '/HTTPEnable/{print $3}')" = "1" ]; then
  PORT="$(scutil --proxy | awk '/HTTPPort/{print $3}')"
  export https_proxy="http://127.0.0.1:$PORT" http_proxy="http://127.0.0.1:$PORT" EI_PROXY="http://127.0.0.1:$PORT"
  echo "① 代理：127.0.0.1:$PORT"
else
  echo "① ⚠️  没检测到系统代理（Veee 没开？）——FactSet / Yahoo 多半连不上，先用本地已有数据重算"
fi

# ---------- 同步 GitHub：Yahoo 快照、上修初期名单记录（early_log.csv）、逐日表现（early_px.csv）只由 GitHub Actions 每天写 ----------
# 拉不下来（没开代理、网络慢、和本地改动冲突）就跳过，用本地已有数据照常生成
echo
if [ -d .git ] && [ "$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" = "main" ]; then
  echo "② 同步 GitHub 上的快照和名单记录"
  BEFORE="$(git rev-parse -q --verify origin/main)"
  if GIT_TERMINAL_PROMPT=0 GIT_HTTP_LOW_SPEED_LIMIT=1000 GIT_HTTP_LOW_SPEED_TIME=20 \
     git pull --rebase --autostash -q origin main >pull_err.log 2>&1; then
    # 本地没提交的改动和拉下来的撞了：这些文件先用 GitHub 上的版本（下面重算会重新生成），本地改动留在 git stash 里
    CONFLICT="$(git diff --name-only --diff-filter=U)"
    if [ -n "$CONFLICT" ]; then
      echo "$CONFLICT" | while read -r f; do git checkout -q HEAD -- "$f"; done
      git reset -q
      echo "   ⚠️  本地未提交的改动和 GitHub 上的冲突，这些文件已换成 GitHub 版本（本地改动存在 git stash 里）："
      echo "$CONFLICT" | sed 's/^/      /'
    fi
    if [ "$BEFORE" = "$(git rev-parse origin/main)" ]; then
      echo "   已是最新"
    else
      echo "   拉下 $(git rev-list --count "$BEFORE"..origin/main) 个提交；最新快照 $(ls yahoo/20*.csv 2>/dev/null | tail -1 | xargs basename 2>/dev/null)"
    fi
  else
    git rebase --abort >/dev/null 2>&1
    echo "   ⚠️  同步失败，用本地已有数据继续：$(grep -v '^\s*$' pull_err.log | tail -1)"
  fi
  rm -f pull_err.log
else
  echo "② 同步 GitHub：不在 main 分支，跳过"
fi

# ---------- 抓数据（失败不中断，用已有数据继续）----------
echo
echo "③ 抓新周报（只抓本地还没有的；FactSet 限流时会自动等）"
NEW="$("$PY" fetch_factset.py --new 2>&1 | grep -E " hub|Traceback|Error")"
if [ -n "$NEW" ]; then echo "$NEW" | sed 's/^/   /'; else echo "   没有新周报（本周的可能还没发布，下次双击会再试）"; fi

# 纳指、标普缓存超过 3 小时就会在下面重算时自动重下

# ---------- 重算 ----------
echo
echo "④ 重算"
set -o pipefail
"$PY" parse_factset.py >/dev/null 2>run_err.log         || die "解析周报失败：$(tail -3 run_err.log)"
echo "   周报解析完成（最新一期 $(tail -1 factset_weekly.csv | cut -d, -f1)）"
"$PY" analyze.py >/dev/null 2>>run_err.log              || die "analyze.py 失败：$(tail -3 run_err.log)"
"$PY" realtime.py 2>>run_err.log | tail -1 | sed 's/^/   实时口径：/' || die "realtime.py 失败：$(tail -3 run_err.log)"
"$PY" speed.py >/dev/null 2>>run_err.log                || die "speed.py 失败：$(tail -3 run_err.log)"
"$PY" make_chart.py 2>>run_err.log | grep html | sed 's/^/   生成：/' || die "make_chart.py 失败：$(tail -3 run_err.log)"
set +o pipefail

# ---------- Yahoo 成分股快照：每天一次，历史从 2026-10-02 起靠它积累 ----------
echo
if [ -d .git ]; then
  echo "⑤ Yahoo 成分股快照：建了仓库后改由 GitHub Actions 每个交易日拍（只留一个写入方，免得两边各拍一份冲突），本地跳过"
elif [ -f "yahoo/$(date +%F).csv" ]; then
  echo "⑤ Yahoo 成分股快照：今天已拍过"
else
  echo "⑤ Yahoo 成分股快照（约半分钟）"
  "$PY" yahoo_snapshot.py 2>/dev/null | head -1 | sed 's/^/   /' || echo "   快照失败（不影响页面）"
fi

# ---------- 打开 ----------
echo
[ -f 修正速度与纳指.html ] || die "没有生成 修正速度与纳指.html"
echo "⑥ 打开网页"
[ "$NO_OPEN" = "1" ] || open 修正速度与纳指.html || die "打不开，手动双击同目录的 修正速度与纳指.html"
echo
echo "✅ 完成。此窗口可以关掉了。"
read -n 1 -s -r -t 15 -p "按任意键关闭（15 秒后自动关闭）…"
echo
