"""数据获取层：财报 PDF 下载 + AKShare 财务数据交叉核对。

TODO(第3周):
- 巨潮资讯 PDF 下载: http://www.cninfo.com.cn (搜索接口 + PDF 直链)
- AKShare: ak.stock_financial_report_sina / ak.stock_balance_sheet_by_report_em
  拿到的财务数据用于第4周「validate」节点的交叉核对
- 缓存策略: 按 (股票代码, 报告期) 缓存到 data_cache/
"""
