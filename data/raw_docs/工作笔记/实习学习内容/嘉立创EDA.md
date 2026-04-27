**eext-ai-library-builder** 是一个为嘉立创 EDA 专业版提供的 AI 驱动的智能建库工具。
### 核心功能
1. **符号提取助手（Symbol Extractor）** - 从PDF/图片提取芯片引脚
     - 从 PDF 数据手册自动提取元件信息
    - 智能识别引脚信息、封装参数和规格
    - 支持布局优化
2. **封装生成器（Footprint Generator）** - 生成各种封装类型
	   - 支持 BGA、DIP、QFN、QFP、SOP 等多种封装类型
    - 从技术文档自动提取参数
    - 实时预览和手动调整

  **symbol.html 工作流**：

```
1. 用户上传PDF/图片 → 
2. AI搜索符号位置 → 
3. 用户手动框选或自动检测 →
4. AI提取引脚信息(pin number, pin name) → 
5. 生成JSON格式的引脚数据 → 
6. 调用EDA API创建符号
```

#### **符号创建流程**：

```
PDF文件
  ↓
[PDF.js解析] → Canvas图片
  ↓
[手动/自动框选] → 坐标信息
  ↓
[AI调用（Vision Model）] → 提取结果JSON
  ├── pinNumber (引脚号)
  ├── pinName (引脚名)
  ├── pinType (引脚类型)
  └── ...
  ↓
[用户编辑/确认] → 参数表格
  ↓
[createSymbol()] → 调用EDA API
  └── eda.sch_PrimitivePolygon.create() [创建边框]
  └── eda.sch_PrimitivePin.create() [创建引脚]
  ↓
 符号创建完成
```
