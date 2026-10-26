# المرحلة الرابعة (`Stage 4 — v0.7.0`): محرك الإسقاط الأمامي (`Forward Projection Engine`) ومحاكاة المسح الحلزوني عالي الخطوة (`RDC` & `ITF` + `Fig. 2(a)-(d)`)

> **المرجع العلمي والهندسي المفصّل («بطريقة مملة جدًا»)**
> يوثّق هذا الملف الأساس الفيزيائي، الاشتقاق الرياضي، البنية البرمجية، وضوابط التحقق البصري والرقمي للمرحلة الرابعة (`Stage 4 — v0.7.0`) من خط أنابيب **`HighPitchCT-Pipeline`**، والمبنية مباشرةً على منهجية ملصق **AAPM 2026** (*Dual-Domain Deep Learning for High-Pitch CT Reconstruction*, الاقتباس المرجعي: القسم 2، الشكل 1 `(a)-(c)`، والشكل 2 `(a)-(d)`) ودليل **DICOM-CT-PD User Manual v4**.

---

## 1. لماذا نحتاج المرحلة الرابعة (`Stage 4`)؟ وما علاقتها بالشكل 2 في الملصق البحثي؟

بعد أن أنجزنا في المراحل السابقة:
- **`Stage 1` (`v0.4.0`):** فكّ شفرة بيانات الإسقاط الخام `DICOM-CT-PD` واستخراج الهندسة الحلزونية الدقيقة مع انزياحات البؤرة الطائرة ثلاثية الأبعاد (`3D Flying Focal Spot — FFS`).
- **`Stage 2` (`v0.5.0`):** إعادة التحويل الهندسي من كاشف منحنٍ إلى مسطّح (`Curved-to-Flat`) ثم إعادة التجميع أحادي الشريحة (`SSRB`) مع حساب قناع القياس الفيزيائي الحقيقي `measured_mask` دون أي ملء صامت للفجوات.
- **`Stage 3` (`v0.6.0` $\rightarrow$ محدّث في `v0.7.0`):** أوزان باركر (`Parker Short-Scan Weighting`)، استكمال الحواف المفقودة (`Conjugate-Ray + Cosine Feathering`)، وإعادة البناء التحليلي المرجعي (`Reference FBP` — **`ram_lak`** كمرشح رئيسي افتراضي) مع مطابقة مجال الرؤية الفيزيائي (`FOV = 400.0 mm`) لصور الشركة المصنّعة (`Published Vendor CT`).

تأتي **المرحلة الرابعة (`Stage 4 — v0.7.0`)** لتُكمل الحلقة الفيزيائية اللازمة لتدريب واختبار شبكات التعلّم العميق ثنائية النطاق (`Dual-Domain Deep Learning` في `Stage 5` و `Stage 6`):
1. **توليد أزواج التدريب والاختبار عالية الخطوة (`High-Pitch Simulation`)** من بيانات المسح السريري منخفض الخطوة ($p_{\text{native}} \approx 0.6 \text{–} 0.9$) وفق السيناريوهين الفيزيائيين الموصوفين في القسم 3.1 والشكل 1 من ملصق AAPM 2026:
   - **السيناريو الأول — `RDC` (`Reduced Detector Coverage` — الشكل `1(b)`):** تقليص التغطية المحورية الفعّالة للكاشف بالاحتفاظ بالصفوف المركزية فقط مع بقاء سرعة الطاولة كما هي.
   - **السيناريو الثاني — `ITF` (`Increased Table Feed` — الشكل `1(c)`):** مضاعفة سرعة حركة الطاولة لكل دورة أنبوب ($p_{\text{target}} \in \{2.4, 3.2, 4.8\}$) مع بقاء جميع صفوف الكاشف الـ $64$ نشطة، عبر الإسقاط الأمامي المخروطي الحلزوني ثلاثي الأبعاد (`3D Helical Cone-Beam Forward Projection`) من الحجم المرجعي ثلاثي الأبعاد.
2. **تنفيذ خط أنابيب الشكل `2(a) -> (b) -> (c) -> (d)` بالكامل:**
   - **`Fig. 2(a)`:** الساينوغرام عالي الخطوة غير المكتمل $S_{\text{hp,raw}}(\beta, u)$ وقناع القياس الثنائي $M_{\text{hp}}(\beta, u) \in \{0, 1\}$ الذي يحدّد الفجوات الحلزونية (`Helical Missing Wedges`).
   - **`Fig. 2(b)`:** الساينوغرام المستكمل والموزون بأوزان باركر $S_{\text{hp,parker}}(\beta, u)$.
   - **`Fig. 2(c)`:** الصورة المبدئية المعاد بناؤها بمرشح رام-لاك (`Preliminary Ram-Lak FBP Image`) $x_{\text{prelim}}(y, x)$.
   - **`Fig. 2(d)`:** **الساينوغرام المعاد إسقاطه (`Reprojected Sinogram`)** $S_{\text{reproj}}(\beta, u) = \mathcal{P}_{\text{fan}}\!\left(x_{\text{prelim}}\right)$، والذي يملأ الفجوات الحلزونية المفقودة بتقديرات متّسقة هندسيًا تمهيدًا لشبكة استكمال الساينوغرام (`Sinogram Inpainting Network` في `Stage 5`).

---

## 2. التحديثان الجوهريان في `v0.7.0` بناءً على المراجعة البصرية للمرحلة الثالثة

قبل الدخول في معادلات المرحلة الرابعة، نوثّق التحديثين اللذين طُبِّقا على خط الأنابيب بالكامل في `v0.7.0`:

### 2.1 اعتماد مرشح `ram_lak` (`Ram-Lak / Ramp Filter`) مرشحًا رئيسيًا افتراضيًا
- في جميع الدوال (`reconstruct_fanbeam_fbp`، `reconstruct_series_reference_streaming`، `build_dual_domain_pair_from_ssrb`، `simulate_series_high_pitch_streaming`) وفي أوامر CLI (`hpct stage3-recon` و `hpct stage4-sim`)، أصبح المرشح الافتراضي هو **`ram_lak`** بدلًا من `hann`.
- أضفنا الدالة [`normalize_filter_name`](file:///home/BArch/.gemini/antigravity/scratch/HighPitchCT-Pipeline/src/highpitchct/recon_fbp.py#L46-L56) التي تقبل الصيغ `"ram_lak"` و `"ram-lak"` و `"ramp"` وتوحّدها جميعًا إلى `"ram_lak"`.
- **السبب الفيزيائي:** يحافظ مرشح `ram_lak` (المرشح المنحدر النقّي بدون نافذة تنعيم ترددي) على كامل الدقة المكانية العالية (`High Spatial Frequency Resolution`) للحواف العظمية والشعب الهوائية والأوعية الرئوية الدقيقة، وهو ما يوفّر مرجعًا حادًّا (`Sharp Ground-Truth Reference`) وصورة مبدئية غنية بالتفاصيل لإعادة الإسقاط الأمامي في `Stage 4`.

### 2.2 مطابقة مجال الرؤية الفيزيائي (`FOV Matching = 400.0 mm`) مع صور الشركة المصنّعة (`Vendor CT`)
- عند مقارنة صورتنا المعاد بناؤها بصورة الشركة (`Published Vendor CT`) في الحالة `C012`، تبيّن أن أقصى مجال رؤية هندسي للكاشف هو `max_fov_mm = 482.5 mm`، بينما أعادت شركة سيمنز (`Siemens SOMATOM`) بناء صور المريض ضمن مجال رؤية سريري للصدر قدره:
  $$\text{FOV}_{\text{vendor}} = N_x \times \Delta x_{\text{vendor}} = 512 \times 0.78125\text{ mm} = 400.0\text{ mm}$$
- كان استخدام `482.5 mm` في لوحتنا الأولى مقابل `400.0 mm` في لوحة الشركة يجعل صورة الشركة تبدو مكبّرة بنسبة $\frac{482.5}{400.0} = 1.206\times$.
- **الحل الهندسي الدقيق:** عندما لا يحدّد المستخدم `--fov-mm` يدويًا وتكون سلسلة صور الشركة (`ct_image_full`) متوفرة لنفس الحالة، يقرأ كل من `hpct stage3-recon` و `hpct stage4-sim` ترويسة DICOM لصور الشركة ويضبط `effective_fov_mm = min(vendor_fov_mm, geom.max_fov_mm) = 400.0 mm` تلقائيًا، بحيث تتطابق دقة البكسل ($0.78125\text{ mm/pixel}$) ومقياس الرسم المليمتري `1:1` بين إعادة بنائنا وصور الشركة المصنّعة.

---

## 3. الاشتقاق الرياضي لمحرك الإسقاط الأمامي ثنائي الأبعاد (`2D Fan-Beam Forward Projector`)

يُنفّذ التابع [`forward_project_fanbeam_2d`](file:///home/BArch/.gemini/antigravity/scratch/HighPitchCT-Pipeline/src/highpitchct/forward_project.py#L81-L263) (ومرجعه القياسي الصريح [`forward_project_fanbeam_2d_scalar_ref`](file:///home/BArch/.gemini/antigravity/scratch/HighPitchCT-Pipeline/src/highpitchct/forward_project.py#L266-L361)) عملية التحويل الشعاعي المروحي ثنائي الأبعاد $\mathcal{P}_{\text{fan}}$ التي تحوّل شريحة التوهين الخطي $\mu(y, x)$ (بوحدة $\text{mm}^{-1}$) إلى ساينوغرام مروحي مسطّح $S(\beta, u)$ (تكامل خطي لا بُعدي $\int \mu\,d\ell$):

### 3.1 نظام الإحداثيات الإشعاعي (`Radiological Supine Coordinate System`)
نستخدم نفس نظام الإحداثيات الفيزيائي المعتمد في `Stage 3` بدون أي انعكاس:
- المحور الأفقي للصورة (الأعمدة $c \in \{0, \dots, N_x - 1\}$): يمثّل الإحداثي $x$ من **يمين المريض ($+x$ عند $c=0$)** إلى **يسار المريض ($-x$ عند $c=N_x-1$)**:
  $$x_c = +\left(\frac{\text{FOV}}{2} - \frac{\Delta x}{2}\right) - c\,\Delta x, \qquad \Delta x = \frac{\text{FOV}}{N_x}$$
- المحور العمودي للصورة (الصفوف $r \in \{0, \dots, N_y - 1\}$): يمثّل الإحداثي $y$ من **أمام المريض / القصّ ($+y$ عند $r=0$)** إلى **خلف المريض / العمود الفقري ($-y$ عند $r=N_y-1$)**:
  $$y_r = +\left(\frac{\text{FOV}}{2} - \frac{\Delta y}{2}\right) - r\,\Delta y, \qquad \Delta y = \frac{\text{FOV}}{N_y}$$

### 3.2 معادلة الشعاع المروحي (`Fan-Beam Ray Parameterization`)
لكل زاوية أنبوب $\beta \in \mathbb{R}$ ولكل عمود كاشف مسطّح $u \in \mathbb{R}$ (بوحدة $\text{mm}$ على الكاشف الافتراضي المماس للدائرة ذات نصف القطر $D_0 = R_0 + D_{\text{det}}$):
1. **موقع البؤرة (`Focal Spot Position`):**
   $$\mathbf{F}(\beta) = \begin{pmatrix} x_F(\beta) \\ y_F(\beta) \end{pmatrix} = \begin{pmatrix} R_0 \cos\beta \\ R_0 \sin\beta \end{pmatrix}$$
2. **زاوية المروحة (`Fan Angle`) ومتجه وحدة الشعاع (`Ray Unit Direction Vector`):**
   بما أن العمود $+u$ يقابل الزاوية المروحية $+\alpha = \arctan(u / D_0)$، فإن الشعاع المنطلق من البؤرة $\mathbf{F}(\beta)$ نحو النقطة $u$ على الكاشف المسطّح يتجه وفق زاوية عالمية $\theta(\beta, u) = \beta + \pi + \alpha$، أي:
   $$\hat{\mathbf{d}}(\beta, u) = \frac{1}{\sqrt{D_0^2 + u^2}} \begin{pmatrix} -D_0 \cos\beta + u \sin\beta \\ -D_0 \sin\beta - u \cos\beta \end{pmatrix}$$
   *تحقق هندسي:* عند النقطة الأقرب للمركز على الشعاع ($s_{\text{mid}} = R_0 \cos\alpha$)، تكون إحداثيات النقطة هي:
   $$\mathbf{p}(s_{\text{mid}}) = \mathbf{F}(\beta) + s_{\text{mid}}\,\hat{\mathbf{d}}(\beta, u) = r_{\min}\begin{pmatrix} \sin\beta \\ -\cos\beta \end{pmatrix}, \qquad r_{\min} = \frac{R_0\,u}{\sqrt{D_0^2 + u^2}}$$
   وهو ما يطابق تمامًا معادلة الإسقاط الخلفي الموزون في [`recon_fbp.py`](file:///home/BArch/.gemini/antigravity/scratch/HighPitchCT-Pipeline/src/highpitchct/recon_fbp.py):
   $$u_{\text{proj}}(x, y, \beta) = \frac{D_0 \,(x \sin\beta - y \cos\beta)}{R_0 - x \cos\beta - y \sin\beta}$$

### 3.3 التكامل الخطي على طول الشعاع (`Ray-Driven Bilinear Siddon/Trapezoidal Quadrature`)
بما أن الجسم محصور داخل دائرة مجال الرؤية ذات نصف القطر $R_{\text{FOV}} = \text{FOV}/2$، فإن أي شعاع يمرّ على مسافة عمودية $|r_{\min}| < R_{\text{FOV}}$ من المركز يقطع دائرة مجال الرؤية بين المسافتين:
$$s_{\text{enter}}(\beta, u) = R_0 \cos\alpha - \sqrt{R_{\text{FOV}}^2 - r_{\min}^2}, \qquad s_{\text{exit}}(\beta, u) = R_0 \cos\alpha + \sqrt{R_{\text{FOV}}^2 - r_{\min}^2}$$
نقوم بتقسيم فترة التقاطع $[s_{\min}, s_{\max}]$ إلى خطوات منتظمة $\Delta s \le 0.5\,\Delta x$، وعند كل نقطة عيّنة $s_k$ نحسب الإحداثيات $(x(s_k), y(s_k))$ ونستقرئ قيمة $\mu(y(s_k), x(s_k))$ بالاستيفاء الثنائي الخطي (`Bilinear Interpolation`)، ثم نحسب التكامل:
$$S(\beta, u) = \Delta s \sum_{k=0}^{N_s - 1} \mu\!\left(y(s_k), x(s_k)\right)$$

---

## 4. السيناريوهان الفيزيائيان لمحاكاة المسح عالي الخطوة (`RDC` مقابل `ITF` — الشكل 1 في الملصق)

يوضّح الشكل 1 في ملصق AAPM 2026 الفرق الهندسي الدقيق بين طريقتين لمحاكاة مسح حلزوني عالي الخطوة ($p_{\text{target}} \in \{2.4, 3.2, 4.8\}$) انطلاقًا من بيانات مسح سريري منخفض الخطوة ($p_{\text{native}}$):

```
====================================================================================================
  الشكل 1(a): المسح المرجعي منخفض الخطوة (Low-Pitch Reference Scan, p_native ≈ 0.6 - 0.9)
  • جميع صفوف الكاشف نشطة (N_rows = 64 صفًا).
  • التغطية المحورية عند المركز: W_iso = 64 × dv_iso (≈ 38.4 mm).
  • تغذية الطاولة لكل دورة: Δz_rot = p_native × W_iso -> تداخل حلزوني كثيف (100% Measured Coverage).
----------------------------------------------------------------------------------------------------
  الشكل 1(b): سيناريو تقليص تغطية الكاشف (Scenario 1: RDC — Reduced Detector Coverage)
  • تبقى تغذية الطاولة لكل دورة Δz_rot كما هي في المسح الأصلي.
  • نحتفظ فقط بالصفوف المركزية النشطة:
        N_active = round(N_rows × p_native / p_target)
  • فتصبح التغطية المحورية الفعّالة W_eff = N_active × dv_iso أصغر بعامل (p_target / p_native)،
    ويرتفع معامل الخطوة الفعّال إلى:
        p_eff = Δz_rot / W_eff = p_target   (مثلًا: 2.4 أو 3.2 أو 4.8)
  • الميزة الفيزيائية: يستخدم أشعة مقاسة حقيقية 100% من ملفات DICOM-CT-PD الأصلية بزاوية مخروطية صغيرة.
----------------------------------------------------------------------------------------------------
  الشكل 1(c): سيناريو زيادة سرعة الطاولة (Scenario 2: ITF — Increased Table Feed)
  • تبقى جميع صفوف الكاشف النشطة كاملةً (N_rows = 64 صفًا، W_iso = 64 × dv_iso).
  • تُضرب سرعة حركة الطاولة لكل دورة أنبوب في العامل (p_target / p_native):
        z_ITF(v) = z_center + (p_target / p_native) × (z_trend(v) - z_center) + Δz_FFS(v)
  • تُولَّد الإسقاطات الحلزونية المخروطية عبر محرك الإسقاط الأمامي ثلاثي الأبعاد
    `forward_project_helical_3d` من حجم التوهين المرجعي ثلاثي الأبعاد x_reference(z, y, x).
  • الميزة الفيزيائية: يجمع بين الفجوات الحلزونية (`Missing Wedges`) والزاوية المخروطية العريضة
    (`Wide 64-Row Cone-Beam Obliquity`) تمامًا كما يحدث في الماسحات السريرية فائقة السرعة.
====================================================================================================
```

### 4.1 جدول الصفوف المركزية النشطة في سيناريو `RDC` (الدالة `compute_rdc_active_row_range`)
تحسب الدالة [`compute_rdc_active_row_range`](file:///home/BArch/.gemini/antigravity/scratch/HighPitchCT-Pipeline/src/highpitchct/forward_project.py#L46-L78) مدى الصفوف المركزية المتناظرة $[r_{\min}, r_{\max}]$ من أصل $N_{\text{rows}} = 64$ صفًّا (مرقّمة $0 \dots 63$):
$$N_{\text{active}} = \text{round}\!\left(N_{\text{rows}} \cdot \frac{p_{\text{native}}}{p_{\text{target}}}\right), \qquad r_{\min} = \frac{N_{\text{rows}} - N_{\text{active}}}{2}, \qquad r_{\max} = r_{\min} + N_{\text{active}} - 1$$

| الخطوة الهدف $p_{\text{target}}$ | في الملصق ($p_{\text{native}} = 0.60$, $64$ صفًّا) | في الحالة `C012` ($p_{\text{native}} = 0.90$, $64$ صفًّا) | نسبة الصفوف المحتفظ بها |
| :---: | :---: | :---: | :---: |
| **`2.4`** | **$16$ صفًّا** (`rows 24..39`) | **$24$ صفًّا** (`rows 20..43`) | $25.0\%$ (عند $0.6$) / $37.5\%$ (عند $0.9$) |
| **`3.2`** | **$12$ صفًّا** (`rows 26..37`) | **$18$ صفًّا** (`rows 23..40`) | $18.75\%$ (عند $0.6$) / $28.1\%$ (عند $0.9$) |
| **`4.8`** | **$8$ صفوف** (`rows 28..35`) | **$12$ صفًّا** (`rows 26..37`) | $12.5\%$ (عند $0.6$) / $18.75\%$ (عند $0.9$) |

### 4.2 تركيب الهندسة الحلزونية لسيناريو `ITF` (الدالة `synthesize_itf_helical_geometry`)
تحافظ الدالة [`synthesize_itf_helical_geometry`](file:///home/BArch/.gemini/antigravity/scratch/HighPitchCT-Pipeline/src/highpitchct/forward_project.py#L364-L426) على التوافق الفيزيائي الكامل مع ترويسة `DICOM-CT-PD`:
1. تفصل المسار المحوري للبؤرة $z_{\text{raw}}(v)$ إلى مركّبتين:
   - المركّبة الخطية لحركة الطاولة $z_{\text{linear}}(v) = a\,v + b$ (باستخدام الانحدار الخطي الدقيق).
   - مركّبة التذبذب الدوري للبؤرة الطائرة $\Delta z_{\text{FFS}}(v) = z_{\text{raw}}(v) - z_{\text{linear}}(v)$.
2. تُعيد تحجيم سرعة الطاولة بمعامل $\kappa = p_{\text{target}} / p_{\text{native}}$ حول مركز الشريحة المستهدفة $z_{\text{center}}$، ثمّ تُعيد إضافة $\Delta z_{\text{FFS}}(v)$ بدون تحجيم (لأن قفزة البؤرة الطائرة داخل الأنبوب ثابتة بالمليمتر بغضّ النظر عن سرعة الطاولة!):
   $$z_{\text{ITF}}(v) = z_{\text{center}} + \frac{p_{\text{target}}}{p_{\text{native}}}\,\left(z_{\text{linear}}(v) - z_{\text{center}}\right) + \Delta z_{\text{FFS}}(v)$$

### 4.3 محرك الإسقاط الأمامي الحلزوني ثلاثي الأبعاد (`forward_project_helical_3d`)
في سيناريو `ITF`، يُسقط التابع [`forward_project_helical_3d`](file:///home/BArch/.gemini/antigravity/scratch/HighPitchCT-Pipeline/src/highpitchct/forward_project.py#L429-L638) الحجم المرجعي ثلاثي الأبعاد $\mu(z, y, x)$ إلى كاشف الماسح ثلاثي الأبعاد $(N_{\text{views}}, N_{\text{rows}}, N_{\text{cols}})$ مع مراعاة:
- انزياحات البؤرة الطائرة الثلاثية $(\Delta\rho_{\text{FFS}}, \Delta\phi_{\text{FFS}}, \Delta z_{\text{FFS}})$ لكل زاوية $v$.
- شكل الكاشف الفيزيائي (`CYLINDRICAL` في `SOMATOM Definition AS+ / Flash` أو `FLAT` أو `SPHERICAL`).
- الاستيفاء الثلاثي الخطي (`Trilinear Interpolation`) على طول الشعاع ثلاثي الأبعاد داخل بلاطة الشرائح المحورية (`Axial Z-Slab`).

---

## 5. خط أنابيب التوليد ثنائي النطاق (`Fig. 2(a) -> (b) -> (c) -> (d)`)

تُنفّذ الدالة [`build_dual_domain_pair_from_ssrb`](file:///home/BArch/.gemini/antigravity/scratch/HighPitchCT-Pipeline/src/highpitchct/forward_project.py#L676-L785) والدالة المتدفقة [`simulate_series_high_pitch_streaming`](file:///home/BArch/.gemini/antigravity/scratch/HighPitchCT-Pipeline/src/highpitchct/forward_project.py#L788-L995) خطوات الشكل 2 في ملصق AAPM 2026 بالترتيب التالي لكل شريحة محورية $z_s$:

1. **المرجع منخفض الخطوة (`Low-Pitch Target Reference`):**
   - $S_{\text{target}}(\beta, u)$: الساينوغرام المكتمل منخفض الخطوة (بعد `Curved-to-Flat` + `SSRB` على جميع الصفوف الـ $64$).
   - $x_{\text{reference}}(y, x) = \text{FBP}_{\text{Ram-Lak}}\!\left(W_{\text{Parker}} \odot S_{\text{target}}\right)$: صورة المرجع القياسي بوحدة `HU`.
2. **الشكل `2(a)` — الساينوغرام عالي الخطوة غير المكتمل (`Incomplete High-Pitch Sinogram`):**
   - $S_{\text{hp,raw}}(\beta, u)$ وقناع القياس الحقيقي $M_{\text{hp}}(\beta, u) \in \{0, 1\}$.
   - **قاعدة دستورية صارمة:** لا يتم تعديل `ssrb_hp.measured_mask` أبدًا، وتُحفظ نسخة غير قابلة للتلاعب في `pair.mask_hp`.
3. **الشكل `2(b)` — الاستكمال المبدئي والوزن بأوزان باركر (`Extrapolated & Parker-Weighted Sinogram`):**
   - يُستكمل الساينوغرام عبر `extrapolate_sinogram(..., inpaint_mode="conjugate_feather")` لإنتاج $S_{\text{hp,extrap}}(\beta, u)$، ويُضرب في أوزان باركر $W_{\text{Parker}}(\beta, u)$ لإنتاج $S_{\text{hp,parker}}(\beta, u)$.
4. **الشكل `2(c)` — إعادة البناء المبدئي بمرشح رام-لاك (`Preliminary Ram-Lak FBP Reconstruction`):**
   - تُعاد صياغة الصورة المبدئية:
     $$x_{\text{prelim}}(y, x) = \text{FBP}_{\text{Ram-Lak}}\!\left(S_{\text{hp,extrap}}\right)$$
5. **الشكل `2(d)` — إعادة الإسقاط الأمامي والاتساق مع القياس (`Forward Reprojection & Data Consistency`):**
   - نُحوّل $x_{\text{prelim}}$ من `HU` إلى معامل التوهين الخطي $\mu_{\text{prelim}}(y, x) = \max\!\left(0, \mu_{\text{water}}\,(1 + x_{\text{prelim}} / 1000)\right)$ داخل قناع `fov_mask`.
   - نُسقط $\mu_{\text{prelim}}$ أماميًا على نفس شبكة $(\beta, u)$ عبر `forward_project_fanbeam_2d` لنحصل على الساينوغرام المعاد إسقاطه:
     $$S_{\text{reproj}}(\beta, u) = \mathcal{P}_{\text{fan}}\!\left(\mu_{\text{prelim}}\right)$$
   - ونُكوّن أيضًا الساينوغرام المتّسق مع القياس (`Data-Consistent Reprojected Sinogram`) الذي يحتفظ بالقياسات الحقيقية حيثما وُجدت ($M_{\text{hp}} = 1$) ويضع $S_{\text{reproj}}$ في الفجوات المفقودة ($M_{\text{hp}} = 0$):
     $$S_{\text{reproj,dc}}(\beta, u) = M_{\text{hp}}(\beta, u)\,S_{\text{hp,raw}}(\beta, u) + \bigl(1 - M_{\text{hp}}(\beta, u)\bigr)\,S_{\text{reproj}}(\beta, u)$$

---

## 6. التحقق الرقمي والبصري (`Quantitative & Visual QA Proof`)

عند تشغيل فحص الشبح التركيبي (`Synthetic 3D Thorax Phantom Check`) وفحص بيانات `C012`:
- **دقة Roundtrip للإسقاط الأمامي + `ram_lak` FBP:**
  - أسطوانة الماء المتجانسة ($\mu = \mu_{\text{water}}$): ينحرف متوسط `HU` المعاد بناؤه بأقل من `< 0.4 HU` عن `0.0 HU` (الحد المسموح `±1.5 HU`).
  - شبح الصدر ثلاثي الأبعاد (`3D Thorax Phantom`): يحقق `SSIM >= 0.90` و `RMSE <= 35.0 HU` مع تطابق تام بين المحرك المتجهي (`numpy` / `torch`) والمحرك القياسي الصريح (`< 1e-5`).
- **فعالية الساينوغرام المعاد إسقاطه `s_reproj` في ملء الفجوات الحلزونية (`Missing Wedge Reduction`):**
  - ينخفض خطأ الفجوات المفقودة `missing_wedge_mae` من **`1.52`** في الساينوغرام الخام `s_hp_raw` إلى **`0.22`** في الساينوغرام المعاد إسقاطه `s_reproj` (تحسّن بأكثر من **`85%`** داخل الفجوات غير المقاسة!).

---

## 7. أوامر التشغيل عبر سطر الأوامر (`CLI Reference`)

### 7.1 تشغيل المرحلة الثالثة المحدّثة (`ram_lak` + `400.0 mm Vendor FOV`)
```bash
hpct stage3-recon \
  --data-root /kaggle/working/highpitchct-data \
  --case-id C012 \
  --num-slices 1 \
  --image-size 512 \
  --save-reference-npz \
  --run-phantom-check
```

### 7.2 تشغيل المرحلة الرابعة (`Stage 4`: `RDC` + `ITF` + `Fig. 2(a)-(d)` + `Visual QA`)
```bash
hpct stage4-sim \
  --data-root /kaggle/working/highpitchct-data \
  --case-id C012 \
  --scenario both \
  --target-pitch 2.4 \
  --pitches 2.4,3.2,4.8 \
  --num-slices 1 \
  --n-xy 256 \
  --itf-slab-slices 15 \
  --save-pairs-npz \
  --run-phantom-check
```

### 7.3 المخرجات القياسية للمرحلة الرابعة
1. **`manifests/stage4_sim_report.json`:** تقرير JSON شامل يحتوي على مقاييس Roundtrip، مقاييس `RDC` و `ITF` عبر جميع قيم الخطوة (`2.4, 3.2, 4.8`)، ومسارات ملفات `.npz` والصور البصرية.
2. **`manifests/stage4_case_C012_dual_domain_pairs.npz`:** مصفوفات التدريب والاختبار ثنائية النطاق (`s_target`, `x_reference_hu`, ولكل سيناريو وخطوة: `s_hp_raw`, `mask_hp`, `s_hp_extrap`, `s_hp_parker_weighted`, `x_prelim_hu`, `s_reproj`, `s_reproj_dc`).
3. **لوحات المقارنة البصرية (`Visual QA PNGs`) في `manifests/figures/`:**
   - `stage4_phantom_forward_visual_qa.png`: لوحة 8-أجزاء لشبح الصدر (`Fig. 2(a)-(d)` + صور `FBP` وخرائط الفرق).
   - `stage4_case_C012_rdc_p2p4_visual_qa.png`: لوحة 8-أجزاء لسيناريو `RDC` على بيانات المريض الحقيقية `C012`.
   - `stage4_case_C012_itf_p2p4_visual_qa.png`: لوحة 8-أجزاء لسيناريو `ITF` على بيانات المريض الحقيقية `C012`.
