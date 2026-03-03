# BEMD-Python - Project Structure

## ディレクトリ構成

```
BEMD_Python/
├── bemd/                    # ライブラリ (Article I)
│   ├── __init__.py          # パッケージエントリ
│   ├── core.py              # bemd() メイン関数
│   ├── sift.py              # sift() シフティング
│   ├── extrema.py           # extrema() 1D極値検出
│   ├── extrema2.py          # extrema2() 2D極値検出
│   └── gridfit.py           # gridfit() 曲面フィッティング
├── cli.py                   # CLI エントリポイント (Article II)
├── tests/
│   └── test_bemd.py         # ユニットテスト 14件 (Article III)
├── examples/
│   └── sample_test.py       # サンプル評価スクリプト
├── bemd.m                   # 移植元 MATLAB ソース
├── bemd_output/             # 出力ディレクトリ (生成物)
├── steering/                # プロジェクトメモリ (Article IV)
│   ├── rules/               # 憲法・ルール
│   ├── product.md           # プロダクトコンテキスト
│   ├── tech.md              # 技術スタック
│   └── structure.md         # 構造定義（本ファイル）
├── storage/                 # データストレージ
│   ├── specs/               # 仕様書
│   ├── archive/             # アーカイブ
│   └── changes/             # 変更履歴
└── musubix.config.json      # MUSUBIX 設定ファイル
```

---

**更新日**: 2026-03-03
