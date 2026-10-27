# المرحلة الخامسة (Stage 5): التعلّم العميق ثنائي المجال المشروط فيزيائياً (Physics-Conditioned Dual-Domain Deep Learning)

تُنفّذ هذه المرحلة النموذج الهجين ثنائي المجال (**Dual-Domain Deep Learning Model**) لإعادة بناء مقاطع الأشعة المقطعية الحلزونية عالية الخطوة (**High-Pitch Helical CT**) استناداً إلى ملصق **AAPM 2026**، مع اعتماد البروتوكول الفيزيائي والسريري المتفق عليه:

1. **التوزيع الفيزيائي الهجين للبيانات (`60% ITF + 40% RDC`)**:
   - **`60% ITF (Increased Table Feed)`**: المحاكاة الفيزيائية الحقيقية لما يحدث في المستشفى عند رفع سرعة حركة الطاولة مع إبقاء جميع صفوف الكاشف (`64 rows`) مفتوحة، مما يلتقط بدقة تشوّه الـ Cone-Beam الحلزوني ثلاثي الأبعاد.
   - **`40% RDC (Reduced Detector Coverage)`**: تقليص صفوف الكاشف الفعّالة على قياسات المريض الحقيقية من `DICOM-CT-PD`، مما يحافظ على الضجيج الفوتوني والإلكتروني الحقيقي (`Real Quantum & Electronic Noise`) داخل الأشعة المقاسة.
2. **المنهج السريري متعدد الخطوات (`Multi-Pitch Clinical Curriculum: p ∈ {1.5, 2.0, 2.5}`)**:
   - التدريب والتقييم يتمّان على قيم الـ Pitch السريرية الواقعية:
     - $p = 1.5$: بداية الـ High-Pitch السريري القياسي (نسبة الفجوة الزاوية $\approx 15\%$).
     - $p = 2.0$: الـ High-Pitch السريري السريع للصدر والطوارئ (نسبة الفجوة الزاوية $\approx 33\%$).
     - $p = 2.5$: الحد السريري الأقصى للـ Single-Source High-Pitch (نسبة الفجوة الزاوية $\approx 46\%$).
3. **التكيّف التلقائي مع أي `native_pitch` للمريض**:
   - سواء كان فحص المريض الأصلي في `DICOM-CT-PD` ذا `native_pitch = 0.65` أو `0.80` أو `0.90`، تبقى قيم الهدف السريرية ثابتة عند $\{1.5, 2.0, 2.5\}$؛ وتقوم دالتا [`synthesize_itf_helical_geometry`](../src/highpitchct/forward_project.py) و[`compute_rdc_active_row_range`](../src/highpitchct/forward_project.py) بحساب سرعة الطاولة وعدد الصفوف الفعّالة تلقائياً لكل مريض.

---

## 1. المعمارية الفيزيائية ثنائية المجال (`DualDomainHighPitchNet`)

تتكوّن الشبكة في [`src/highpitchct/dual_domain_dl.py`](../src/highpitchct/dual_domain_dl.py) من ثلاث ركائز متصلة تفاضلياً (`End-to-End Differentiable`):

### أ) المجال الأول: شبكة إكمال الساينوغرام المشروطة فيزيائياً (`SinoInpaintUNet` + `HardDataConsistencyLayer`)
- **مدخلات الشبكة (4 قنوات، الشّكل `(B, 4, N_beta, N_u)`)**:
  1. `s_hp_raw`: الساينوغرام المربّن عند الـ High-Pitch مع أصفار في الفجوة الزاوية (`Fig. 2(a)`).
  2. `s_reproj`: الساينوغرام المعاد إسقاطه (`Reprojected Sinogram`) من الصورة الأولية (`Fig. 2(d)`).
  3. `mask_hp`: القناع الثنائي الفيزيائي الصريح ($1$ للأشعة المقاسة فعلياً، $0$ للفجوة المفقودة).
  4. `pitch_wedge_cond`: خريطة التكييف الفيزيائي للـ Pitch الفعلي وحجم الفجوة الزاوية.
- **طبقة الاتساق الفيزيائي الصارم (`HardDataConsistencyLayer`)**:
  $$\hat{s}_{\text{DC}}(\beta, u) = M(\beta, u) \cdot s_{\text{hp\_raw}}(\beta, u) + \bigl(1 - M(\beta, u)\bigr) \cdot \max\bigl(0,\, \hat{s}_{\text{pred}}(\beta, u)\bigr)$$
  تضمن هذه الطبقة رياضياً وفيزيائياً أن جميع الأشعة المقاسة فعلياً من الجهاز ($M = 1$) تبقى مطابقة بنسبة $100\%$ للقياس الحقيقي ($\max |s_{\text{DC}} - s_{\text{hp\_raw}}|_{M=1} = 0.0$).

### ب) الجسر الفيزيائي التفاضلي (`DifferentiableParkerFBPBridge`)
- يحوّل الساينوغرام المكتمل $\hat{s}_{\text{DC}}$ مباشرة إلى صورة مقطعية بوحدة `HU` ($\hat{x}_{\text{DC}}$) داخل PyTorch مع تمرير التدرجات (`Gradients`) من مجال الصورة إلى مجال الساينوغرام:
  1. أوزان باركر (`Parker Short-Scan Weighting`) + وزن جيب التمام (`Cosine Obliquity Weighting`).
  2. التمديد الجانبي بجيب التمام المرفوع (`Raised-Cosine Lateral Taper`) لمنع التفاف FFT.
  3. الضرب في استجابة التردد لفلتر **`ram_lak`** المعاير (`Kak & Slaney Spatial Kernel`).
  4. الإسقاط الخلفي المروحي التفاضلي (`Differentiable Fan-Beam Backprojection`) والتحويل إلى `HU`.

### ج) المجال الثاني: شبكة تنقية الصورة المقطعية (`ImageRefineUNet`)
- **مدخلات الشبكة (3 قنوات، الشكل `(B, 3, H, W)`)**:
  1. `x_dc_norm`: الصورة المقطعية الخارجة من الجسر الفيزيائي $\hat{x}_{\text{DC}}$ بعد إكمال الساينوغرام.
  2. `x_prelim_norm`: الصورة الأولية قبل الإكمال $x_{\text{prelim}}$ (`Fig. 2(c)`) كمرجع لبصمة الـ Streak Artifacts.
  3. `fov_pitch_map`: قناع دائرة الـ FOV مضروباً في قيمة الـ Pitch الفعلي.
- تتنبأ الشبكة بالتصحيح المتبقي $\Delta x_{\text{HU}}$ داخل دائرة الـ FOV لإزالة الخطوط المتبقية مع الحفاظ على حدة الحواف التشريحية لفلتر **`ram_lak`**.

---

## 2. واجهة سطر الأوامر (`CLI`)

### أ) الفحص الذاتي الكامل على الفانتوم (`Synthetic Phantom Check`)
```bash
hpct stage5-train \
  --data-root /kaggle/working/highpitchct-data \
  --run-phantom-check \
  --pitches "1.5,2.0,2.5" \
  --itf-ratio 0.60 \
  --filter-name ram_lak \
  --backend torch
```

### ب) التدريب والتقييم الهجين (`60% ITF + 40% RDC`) على بيانات مريض حقيقي (`DICOM-CT-PD`)
```bash
hpct stage5-train \
  --data-root /kaggle/working/highpitchct-data \
  --case-id C012 \
  --scenario both \
  --pitches "1.5,2.0,2.5" \
  --itf-ratio 0.60 \
  --fov-mm 400.0 \
  --image-size 256 \
  --epochs 12 \
  --filter-name ram_lak \
  --backend torch \
  --save-model \
  --save-dataset-npz
```

تُحفظ لوحة المقارنة البصرية السداسية في:
`manifests/figures/stage5_<case_id>_dl_visual_qa.png`
وتقرير المقاييس الكمية في:
`manifests/stage5_dl_report.json`
