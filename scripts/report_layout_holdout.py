"""Report paired layout inference; does not change held-out data or model rules."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "outputs/layout-pilot-v1"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = {}
    for row in rows:
        groups.setdefault(row["source_id"], {})[row["template_id"]] = row
    if len(groups) != 12 or any(set(value) != {"A", "D"} for value in groups.values()):
        raise ValueError("Paired evaluation incomplete")
    for group in groups.values():
        if group["A"]["gt"] != group["D"]["gt"]:
            raise ValueError("GT differs between layouts")
    a,d = (summary["by_template"][key] for key in ["A","D"])
    transitions = {"both_correct":0,"A_only_correct":0,"D_only_correct":0,"both_incorrect":0}
    for group in groups.values():
        ca,cd=group['A']['document_correct'],group['D']['document_correct']
        key='both_correct' if ca and cd else 'A_only_correct' if ca else 'D_only_correct' if cd else 'both_incorrect'
        transitions[key]+=1
    evidence=[]
    for source,group in groups.items():
        if group['D']['field_errors']:
            evidence.append({"source_id":source,"A_image":group['A']['image'],"D_image":group['D']['image'],
                             "A_errors":group['A']['field_errors'],"D_errors":group['D']['field_errors'],
                             "A_document_correct":group['A']['document_correct'],"D_document_correct":group['D']['document_correct']})
    (folder/'paired-errors.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 现有微调模型的陌生版式试点','',
           '使用已有 Qwen3-VL-2B QLoRA 适配器直接推理。没有训练、更新模型权重或修改评分器。',
           '', '## 设计与边界','',
           '12 个新生成源单据，各渲染 A/D 两个版本，共 24 张。GT 内容哈希与原 train/val/test 无重合。',
           'A：原渲染器，底部补白 40px。D：序号列移至末列、表头位置变化、合计移到表格上方、取消竖线并弱化横线、增加备注。',
           '每对内容、日期显示值、字体与字号、画布大小完全相同；不加模糊、噪声或旋转。两版推理输入都缩放为 384×384。',
           '模板与提示词在推理前固定，提示词沿用训练清单保存的指令。测试结果未用于改规则或调参。',
           '这是多种布局变化的组合测试，不能把差异单独归因于列顺序或表格线；12 个源单据的结果只作试点，不代表真实单据泛化。',
           '', '## 成绩','', '| 版式 | 字段 F1 | 金额正确/总数 | Schema 合规 | 整单正确 | 推理异常 |',
           '|---|---:|---|---|---|---|']
    for name,s in [('A 原版式',a),('D 陌生版式',d)]:
        lines.append(f"| {name} | {s['f1']:.4f} | {s['amount_correct']}/{s['amount_total']} | {s['schema_valid_count']}/{s['n']} | {s['document_correct_count']}/{s['n']} | {s['inference_failure_count']}/{s['n']} |")
    lines += ['',f"D − A 字段 F1 差值：**{d['f1']-a['f1']:+.4f}**。所有样本保留在分母，异常计失败。",'',
              f"逐源单据整单结果：两版都正确 {transitions['both_correct']}；仅 A 正确 {transitions['A_only_correct']}；仅 D 正确 {transitions['D_only_correct']}；两版都不正确 {transitions['both_incorrect']}。",'',
              '## 运行与时延','',f"GPU：{summary['configuration']['gpu']}；基座缓存版本 `{summary['configuration']['snapshot_revision']}`；4bit，已有 SFT adapter，batch=1，max_new_tokens=768，贪心解码。",
              f"加载耗时 {summary['model_load_seconds']:.1f} 秒；排除加载和一次 32-token 合成图预热后，平均 A {summary['mean_seconds']['A']:.2f} 秒/张，D {summary['mean_seconds']['D']:.2f} 秒/张。",
              '端到端计时包含读图、缩放、processor 编码、生成和解码；前后 CUDA 同步，排除评分和结果文件写入。',
              f"PyTorch CUDA 峰值 allocated {summary['peak_allocated_bytes']/1024**3:.2f} GiB，reserved {summary['peak_reserved_bytes']/1024**3:.2f} GiB，含驻留模型；不是整机 GPU 总占用。",
              '这批图与 OCR 的 102 张基线图不同，不能把两次运行时延直接作为同集速度排名。','',
              '## 配对错误示例','']
    for e in sorted(evidence,key=lambda e:len(e['D_errors']),reverse=True)[:4]:
        lines.append(f"- `{e['source_id']}`：A 错误路径 {len(e['A_errors'])}，D 错误路径 {len(e['D_errors'])}。")
        for err in e['D_errors'][:3]:
            lines.append(f"  `{err['path']}`：GT={err['gt']!r}，D 输出={err['pred']!r}。")
    if not evidence:
        lines.append('D 未发现字段错误；小样本通过不能视为陌生版式问题已解决。')
    lines += ['', '## 产物与复现','',
              '- [数据与模板协议](../data/benchmarks/layout_v1/protocol.json) · [24 张图片清单](../data/benchmarks/layout_v1/manifest.jsonl)',
              '- [配置与模型指纹](../outputs/layout-pilot-v1/configuration.json) · [汇总](../outputs/layout-pilot-v1/summary.json)',
              '- [原始预测与逐样本评分](../outputs/layout-pilot-v1/predictions.jsonl) · [配对差异](../outputs/layout-pilot-v1/paired-errors.json)',
              '', '```powershell','$env:PYTHONUTF8 = "1"',
              '.\\venv-gld\\Scripts\\python.exe scripts\\run_layout_holdout.py',
              '.\\venv-gld\\Scripts\\python.exe scripts\\report_layout_holdout.py','```','',
              '推理脚本仅复用相同配置的缓存，拒绝混用配置。生成数据脚本拒绝覆盖已存在的留出目录。',
              '真实中文单据仍待用户提供可使用的数据目录；项目当前 public 目录中的跨域数据不能冒充同 Schema 的真实材料清单验证。']
    (ROOT/'docs/layout-holdout-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'A_f1':a['f1'],'D_f1':d['f1'],'transitions':transitions},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
