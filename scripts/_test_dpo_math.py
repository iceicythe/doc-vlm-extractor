"""train_dpo.py 的 CPU 单测。

只测"不依赖 unsloth"的两块，但恰好是最容易写错的两块：
  1. seq_logprob_masked 的位置对齐 / 分块 / padding 屏蔽
  2. dpo_loss 的数值与梯度

跑法：venv-gld/Scripts/python.exe scripts/_test_dpo_math.py
（纯 CPU，秒级）
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import train_dpo as T  # noqa: E402  （模块级不 import unsloth，可安全导入）

torch.manual_seed(0)
V, H, B, P, L = 97, 16, 2, 11, 7
TOT = P + L
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


class _Out:
    def __init__(self, h):
        self.hidden_states = (h,)


class _TinyLM(torch.nn.Module):
    """hidden_states 由 token 与位置共同决定，lm_head 就是恒等映射。
    这样任何位置偏移都会改变结果，测得出 off-by-one。"""

    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(H, V, bias=False)
        self.calls = 0

    def forward(self, input_ids=None, attention_mask=None, use_cache=False,
                output_hidden_states=False, **kw):
        self.calls += 1
        b, t = input_ids.shape
        pos = torch.arange(t).float().view(1, t, 1)
        tok = input_ids.float().unsqueeze(-1)
        # 前 H-2 维混入 token/位置信息，其余置 0
        h = torch.zeros(b, t, H)
        h[..., 0] = torch.sin(tok.squeeze(-1) * 0.7)
        h[..., 1] = torch.cos(pos.squeeze(-1) * 1.3)
        return _Out(h)          # _Out 内部包成 tuple，模拟 HF 的 hidden_states


def naive_logprob(model, input_ids, comp_mask, n_prompt):
    """朴素参照实现：先算全量 logits，再按 HF 惯例 shift 后 gather。

    注意 comp_mask 必须一起乘进来，否则和实现里的语义不等价
    （第一版漏了这步，导致虚报失败）。
    """
    h = model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
    logits = model.lm_head(h).float()                      # (B, T, V)
    shift_logits = logits[:, :-1, :]                       # 预测 token 1..T-1
    shift_labels = input_ids[:, 1:]
    lp = F.log_softmax(shift_logits, dim=-1).gather(
        -1, shift_labels.unsqueeze(-1)).squeeze(-1)         # (B, T-1)，对应 token 1..T-1
    lp_tail = lp[:, n_prompt - 1:]                         # (B, T-n_prompt)，对应 token n_prompt..T-1
    return (lp_tail * comp_mask).sum(dim=1)


print("=" * 62)
print("1. seq_logprob_masked 位置对齐 / 分块 / mask")
print("=" * 62)

model = _TinyLM().eval()
ids = torch.randint(0, V, (B, TOT))
ids[:, P:] = torch.tensor([[1, 2, 3, 4, 5, 6, 7],
                           [9, 8, 7, 0, 0, 0, 0]])         # 第二行只有 3 个真 token
cmask = torch.tensor([[1., 1, 1, 1, 1, 1, 1],
                      [1., 1, 1, 0, 0, 0, 0]])
attn = torch.ones(B, TOT)

ref = naive_logprob(model, ids, cmask, P)
with torch.no_grad():
    got128 = T.seq_logprob_masked(model, ids, attn, P, cmask, chunk=128)
    got1 = T.seq_logprob_masked(model, ids, attn, P, cmask, chunk=1)
    got0 = T.seq_logprob_masked(model, ids, attn, P, torch.zeros_like(cmask), chunk=128)

check("与朴素 shift+gather 实现一致", torch.allclose(got128, ref, atol=1e-5),
      f"max|Δ|={(got128-ref).abs().max().item():.2e}")
check("分块大小不影响结果（chunk=1 vs 128）",
      torch.allclose(got128, got1, atol=1e-4),
      f"max|Δ|={(got128-got1).abs().max().item():.2e}")
check("comp_mask 全 0 时返回 0（padding 被屏蔽）",
      got0.abs().max().item() < 1e-9)

# 行2 有 4 个 padding，屏蔽必须真的生效：带 mask 与不带 mask 结果应不同
with torch.no_grad():
    unmasked = T.seq_logprob_masked(model, ids, attn, P, torch.ones_like(cmask), chunk=128)
check("行2的 padding 确实被排除（masked != unmasked）",
      abs(got128[1].item() - unmasked[1].item()) > 1e-6,
      f"masked={got128[1].item():.3f} 全1={unmasked[1].item():.3f}")
check("行1 全为真 token，mask 不影响它",
      abs(got128[0].item() - unmasked[0].item()) < 1e-6)

# off-by-one：把 n_prompt 挪一位，结果必须变（否则说明 n_prompt 没被真正使用）
with torch.no_grad():
    shifted = T.seq_logprob_masked(model, ids, attn, P - 1,
                                   torch.ones(B, TOT - (P - 1)), chunk=128)
check("n_prompt 挪一位会改变结果（无 off-by-one 掩盖）",
      not torch.allclose(got128, shifted, atol=1e-6))

# 形状守卫：mask 长度不对时必须明确报错，而不是广播出诡异结果
try:
    with torch.no_grad():
        T.seq_logprob_masked(model, ids, attn, P, cmask[:, :-1], chunk=128)
    check("comp_mask 长度不符时抛错", False, "没有抛错")
except ValueError as e:
    check("comp_mask 长度不符时抛错", "comp_mask" in str(e))

print()
print("=" * 62)
print("2. dpo_loss 数值与梯度")
print("=" * 62)

zero = torch.zeros(2, requires_grad=True)
loss, acc, rm, rc, rr = T.dpo_loss(zero[0], zero[0], zero[1] * 0, zero[1] * 0, beta=0.1)
check("policy==ref 时 loss == ln2", abs(loss.item() - math.log(2)) < 1e-6,
      f"{loss.item():.6f} vs {math.log(2):.6f}")
check("margin==0 时 acc == 0（无偏好信息）", acc.item() == 0.0)
check("margin==0 时 rewards_chosen == rewards_rejected", abs(rc.item() - rr.item()) < 1e-9)

def analytic(pc_v, pr_v, beta):
    """loss = -logsigmoid(β(pc-pr)) = log(1 + e^{-β(pc-pr)})"""
    return math.log(1 + math.exp(-beta * (pc_v - pr_v)))


zc2 = torch.zeros(2)
pc = torch.tensor([0.0, 0.0])
pr = torch.tensor([-5.0, -5.0])
loss_mid, acc_mid, rm_mid, rc_mid, rr_mid = T.dpo_loss(pc, pr, zc2, zc2, beta=0.1)
check("loss 与解析解 log(1+e^{-βΔ}) 一致",
      abs(loss_mid.item() - analytic(0.0, -5.0, 0.1)) < 1e-6,
      f"{loss_mid.item():.6f} vs {analytic(0.0, -5.0, 0.1):.6f}")
check("chosen 更受偏好时 acc == 1", acc_mid.item() == 1.0)
check("reward_margin == β·(pc-pr) = 0.5", abs(rm_mid.item() - 0.5) < 1e-6,
      f"{rm_mid.item():.4f}")

# 注意 β 的量级：β=0.1 时 5 nat 的 logp 差只折成 0.5 的 margin，
# loss 停在 0.474 —— 所以"DPO loss 训练中一直在 0.69 附近"是正常的，不是没学到
one = torch.zeros(1)
o1 = torch.tensor([0.0])
check("margin 很大时 loss → 0",
      T.dpo_loss(o1, torch.tensor([-20.0]), one, one, beta=1.0)[0].item() < 1e-8,
      f"{T.dpo_loss(o1, torch.tensor([-20.0]), one, one, beta=1.0)[0].item():.2e}")
lo = T.dpo_loss(o1, torch.tensor([-5.0]), one, one, beta=0.5)[0].item()
hi = T.dpo_loss(o1, torch.tensor([-5.0]), one, one, beta=1.0)[0].item()
check("margin 增大 → loss 单调减小", hi < lo, f"β=0.5 {lo:.4f} → β=1.0 {hi:.4f}")
loss_bad = T.dpo_loss(torch.tensor([-5.0]), torch.tensor([0.0]), zc2, zc2, 0.1)[0]
check("chosen 反而更低时 loss > ln2 且 acc == 0",
      loss_bad.item() > math.log(2)
      and T.dpo_loss(torch.tensor([-5.0]), torch.tensor([0.0]), zc2, zc2, 0.1)[1].item() == 0.0,
      f"loss={loss_bad.item():.4f}")

# 梯度方向：想降低 loss，应把 chosen 的 logp 往上推、rejected 往下推
pc_g = torch.zeros(1, requires_grad=True)
pr_g = torch.zeros(1, requires_grad=True)
l, *_ = T.dpo_loss(pc_g, pr_g, torch.zeros(1), torch.zeros(1), beta=0.1)
l.backward()
check("∂loss/∂logp_chosen < 0（提升 chosen）", pc_g.grad.item() < 0,
      f"{pc_g.grad.item():+.4f}")
check("∂loss/∂logp_rejected > 0（压低 rejected）", pr_g.grad.item() > 0,
      f"{pr_g.grad.item():+.4f}")
check("两梯度大小相等（对称性）",
      abs(pc_g.grad.item() + pr_g.grad.item()) < 1e-9)
check("β 缩放：∂loss/∂logp = -β/2（β=0.1 → -0.05）",
      abs(pc_g.grad.item() + 0.05) < 1e-6, f"{pc_g.grad.item():.6f}")

print()
print("=" * 62)
print("3. find_lm_head 逐层剥离")
print("=" * 62)


class _Wrapper(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = inner


w = _Wrapper(_TinyLM())
check("能从 3 层包装里找到 lm_head", T.find_lm_head(w) is w.base_model.model.lm_head)

print()
print("=" * 62)
print("4. extend_mm_token_type_ids 形状补齐（GPU IndexError 的根因）")
print("=" * 62)

# 真实场景：processor 只对 prompt 产出 mm_token_type_ids (B, P)，
# 而 input_ids / attention_mask 已经拼成 (B, P+L) —— get_rope_index 会
# 拿 attention_mask 去索引它，长度不符直接 IndexError。
prompt_mm = torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])       # (B, P)

full_mm = T.extend_mm_token_type_ids(prompt_mm, TOT)
check("补齐后长度 == 序列长度", tuple(full_mm.shape) == (B, TOT),
      f"{tuple(prompt_mm.shape)} -> {tuple(full_mm.shape)}")
check("prompt 段原样保留（图像标记未被破坏）",
      torch.equal(full_mm[:, :P], prompt_mm))
check("completion 段全为 0（纯文本 token）",
      full_mm[:, P:].abs().sum().item() == 0)
check("不修改入参", prompt_mm.shape[-1] == P)

check("长度已正确时原样返回（不复制）",
      T.extend_mm_token_type_ids(prompt_mm, P) is prompt_mm)
check("None 透传", T.extend_mm_token_type_ids(None, TOT) is None)

try:
    T.extend_mm_token_type_ids(prompt_mm, P - 3)
    check("目标长度过短时抛错", False, "没有抛错")
except ValueError as e:
    check("目标长度过短时抛错", "超过" in str(e))

# fail fast：长度不符必须在跑前向之前就报错，而不是让 GPU 前向抛 IndexError
try:
    with torch.no_grad():
        T.seq_logprob_masked(model, ids, attn, P, cmask,
                             {"mm_token_type_ids": prompt_mm}, chunk=128)
    check("mm_token_type_ids 长度不符时提前抛错", False, "没有抛错")
except ValueError as e:
    check("mm_token_type_ids 长度不符时提前抛错", "mm_token_type_ids" in str(e))

with torch.no_grad():
    got_mm = T.seq_logprob_masked(model, ids, attn, P, cmask,
                                  {"mm_token_type_ids": full_mm}, chunk=128)
check("补齐后前向正常，结果与不传该参数一致",
      torch.allclose(got_mm, got128, atol=1e-6),
      f"max|Δ|={(got_mm-got128).abs().max().item():.2e}")

print()
print("=" * 62)
print("5. ref_dispatch_probe：验证 set_adapter 真的切了前向")
print("=" * 62)


class _FakePeft(torch.nn.Module):
    """模拟 PEFT 的多 adapter 结构：lora_A / lora_B 是 ModuleDict，
    key 是 adapter 名、value 是权重模块。ref 与 default 权重逐位相同，
    复刻 DPO「ref 是 policy 副本」的初始状态。"""

    def __init__(self, honor_switch: bool = True):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({
            "default": torch.nn.Linear(4, 4, bias=False),
            "ref": torch.nn.Linear(4, 4, bias=False),
        })
        self.lora_B = torch.nn.ModuleDict({
            "default": torch.nn.Linear(4, 4, bias=False),
            "ref": torch.nn.Linear(4, 4, bias=False),
        })
        with torch.no_grad():
            for k in ("lora_A", "lora_B"):
                getattr(self, k)["ref"].weight.copy_(getattr(self, k)["default"].weight)
        self._active = "default"
        self.honor_switch = honor_switch

    def set_adapter(self, name, inference_mode=False):
        self._active = name

    def probe_value(self) -> float:
        """模拟一次前向：结果只取决于当前 active adapter 的权重。"""
        which = self._active if self.honor_switch else "default"
        return float(self.lora_B[which].weight.sum().item())


def _snap(m):
    return [p.detach().clone() for p in m.parameters()]


fm = _FakePeft()
before = _snap(fm)
base_val = fm.probe_value()                      # active = default


def _probe_fm():
    fm.set_adapter("ref", inference_mode=True)
    try:
        return fm.probe_value()
    finally:
        fm.set_adapter("default")


shifted = T.ref_dispatch_probe(fm, _probe_fm)
check("扰动 ref 权重后结果改变（切换生效）",
      shifted is not None and abs(shifted - base_val) > 1e-6,
      f"{base_val:.4f} -> {shifted:.4f}" if shifted is not None else "返回 None")
check("扰动后权重被精确还原（copy_ 而非乘除往返）",
      all(torch.equal(a, b) for a, b in zip(before, _snap(fm))))
check("还原后结果回到原值", abs(_probe_fm() - base_val) < 1e-9)

# 对照组：切换失效（前向永远读 default）时，探针必须"看不出变化"
fm_bad = _FakePeft(honor_switch=False)
bad_base = fm_bad.probe_value()
bad = T.ref_dispatch_probe(
    fm_bad, lambda: (fm_bad.set_adapter("ref"), fm_bad.probe_value())[1])
check("切换失效时探针返回原值（能被判定为 FAIL）",
      bad is not None and abs(bad - bad_base) < 1e-9, f"{bad:.4f} vs {bad_base:.4f}")

check("无多 adapter 结构时返回 None（而不是抛错）",
      T.ref_dispatch_probe(torch.nn.Linear(4, 4), lambda: 0.0) is None)

print()
if fails:
    print(f"{len(fails)} 项失败: {fails}")
    sys.exit(1)
print("全部通过")
