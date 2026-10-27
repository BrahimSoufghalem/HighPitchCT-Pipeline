# HighPitchCT-Pipeline

إطار بحثي وهندسي محكم وقابل للتدقيق لإعادة بناء وتقييم نموذج **ثنائي المجال (Dual-Domain Deep Learning)** لإعادة بناء التصوير المقطعي الحلزوني مرتفع الـ Pitch (**High-Pitch Helical CT**) انطلاقًا من بيانات الإسقاط الخام في مجموعة **TCIA (`LDCT-and-Projection-data`)**، ومُصمَّم خصيصًا لمذكرة ماستر في **الفيزياء الطبية** وفق نظام بوابات المراحل الصارم (**Stage-Gate**) وبوابة **المقارنة البصرية الإلزامية (`Mandatory Visual QA`)**.

> **الإصدار الحالي:** `v0.8.0` — **المراحل `Stage 0` إلى `Stage 4` مُثبتة فعليًا على المريض الحقيقي `C012` على Kaggle، والمرحلة `Stage 5 (Dual-Domain Deep Learning)` مكتملة ومحققة بـ `52/52` اختبارًا**:
> - اعتماد مرشح **`ram_lak` (`Ram-Lak / Ramp Filter`)** مرشحًا رئيسيًا افتراضيًا في جميع مراحل إعادة البناء للحفاظ على كامل الدقة المكانية العالية.
> - مطابقة مجال الرؤية الفيزيائي (`FOV = 400.0 mm`) تلقائيًا مع صور الشركة المصنّعة (`Published Vendor CT`) لإزالة أي فرق تكبير بصري (`1:1 Millimeter Scale`).
> - محرك الإسقاط الأمامي ثنائي الأبعاد (`forward_project_fanbeam_2d`) وثلاثي الأبعاد (`forward_project_helical_3d` مع `3D FFS`) على `numpy` و`torch` و`torch_cuda` دون أي انتقال صامت.
> - **المرحلة 5 (`Stage 5 — Dual-Domain Deep Learning`):** الشبكة العصبية ثنائية المجال المشروطة فيزيائيًا (`SinoInpaintUNet` بـ 4 قنوات + `HardDataConsistencyLayer` + الجسر التفاضلي `DifferentiableParkerFBPBridge` بمرشح `ram_lak` + `ImageRefineUNet` بـ 3 قنوات) مع بروتوكول التدريب الهجين الواقعي **`60% ITF + 40% RDC`** والمنهج السريري متعدد الخطوات **`p ∈ {1.5, 2.0, 2.5}`** الذي يتكيّف تلقائيًا مع أي `native_pitch` للمريض.

---

## 0. خارطة المراحل الكاملة للمشروع (`Stage 0` $\rightarrow$ `Stage 9`)

| المرحلة | الاسم والمحتوى العلمي والهندسي | الحالة | المخرجات وبوابة القبول |
| :--- | :--- | :---: | :--- |
| **`Stage 0`** | اكتشاف وتنزيل `TCIA` فائق السرعة الواعي ببيئة `Kaggle <= 18 GiB` + فهرسة رؤوس `DICOM-CT-PD v4` | ✅ **مكتملة ومُثبتة على Kaggle (`v0.3.1`)** | `13,446` إسقاطًا + `351` صورة للمريض `C012` بحالة `VALID` (`1.419 GiB`، `pitch = 0.8966`) |
| **`Stage 1`** | الهندسة الحلزونية ثلاثية الأبعاد (`3D FFS`) + قارئ `DICOM-CT-PD` الآمن للذاكرة + 3 براهين توجيه فيزيائية + فانتوم الصدر ثلاثي الأبعاد + `Visual QA` | ✅ **مكتملة ومُثبتة على `C012` في Kaggle (`v0.5.0`)** | `geometry.py`, `ctpd_reader.py`, `phantom.py`, اجتياز البراهين الثلاثة على `C012` في `49.5s` ومعاينة الصور البصرية |
| **`Stage 2`** | إعادة الترتيب الهندسي المتجهي والتدفقي: `Curved-to-Flat` + `SSRB` مع معالجة `3D FFS` وقناع القياس الصريح `measured_mask` ولوحات المقارنة البصرية | ✅ **مكتملة ومُثبتة على `C012` في Kaggle (`v0.5.0` / `v0.6.0`)** | تغطية `100.00%` في `Short-Scan` لـ 5 شرائح صدر من `C012` عند `pitch=0.90`، وتغطية `84.10%` عند `High-Pitch RDC (eff. pitch=1.79)` + إصلاح حدود `Xeon AVX-512` |
| **`Stage 3`** | استكمال الجوانب (`Conjugate-Ray + Feathering`) + ترجيح `Parker` الدقيق للمسح القصير + `Spatial Ram-Lak FBP` وإنتاج المرجع $x_{\text{reference}}$ ومطابقة `FOV = 400.0 mm` | ✅ **مكتملة ومُثبتة على `C012` في Kaggle (`v0.6.0` / `v0.7.0`)** | `recon_fbp.py`, `hpct stage3-recon`, معايرة ماء `0.0 HU`، مرشح `ram_lak` رئيسي، مطابقة `FOV = 400.0 mm` مع صور `Siemens`، ولوحات مقارنة بصرية ذات 6 نوافذ |
| **`Stage 4`** | محرك الإسقاط الأمامي (`2D Fan-Beam` + `3D Helical Cone-Beam`) ومحاكاة سيناريوهات `High-Pitch` (`RDC` الشكل `1(b)` و`ITF` الشكل `1(c)`) وإعادة الإسقاط $S_{\text{reproj}}$ (`Fig. 2(d)`) | ✅ **مكتملة ومُثبتة على `C012` في Kaggle (`v0.7.0`)** | `forward_project.py`, `hpct stage4-sim`, فحص الثبات الذاتي `Forward -> Ram-Lak FBP` (`SSIM >= 0.90`)، خفض خطأ الفجوات المفقودة بأكثر من `85%` عبر $S_{\text{reproj}}$، ولوحات مقارنة بصرية ذات 8 نوافذ |
| **`Stage 5`** | الشبكة العصبية ثنائية المجال المشروطة فيزيائيًا (`SinoNet + Hard DC + Differentiable Parker Ram-Lak FBP Bridge + ImgNet`) والتدريب الهجين (`60% ITF + 40% RDC`) متعدد الخطوات (`p ∈ {1.5, 2.0, 2.5}`) | ✅ **مكتملة برمجيًا (`v0.8.0`) — جاهزة لتشغيل `C012` على Kaggle** | `dual_domain_dl.py`, `hpct stage5-train`, تطابق الجسر التفاضلي `< 0.05 HU`، اتساق فيزيائي صارم `0.0` انحراف على الأشعة المقاسة، ولوحات مقارنة بصرية ذات 6 نوافذ (`stage5_*_dl_visual_qa.png`) |
| **`Stage 6`** | توسيع التدريب متعدد المرضى (`Multi-Patient Split`) مع منع تسرب الشرائح (`Zero Leakage`) | ⏭ **التالية مباشرةً** | تغليف عينات التدريب والتحقق والاختبار عبر عدة مرضى من `TCIA` |
| **`Stage 7`** | مصفوفة التقييم الكمي الشامل ودراسات الحذف (`Ablations`: `FBP` مقابل `Sino-only` مقابل `Image-only` مقابل `Dual-Domain`) | ⏳ قادمة | جداول ومنحنيات `RMSE [HU]`, `PSNR`, `SSIM`, وبروفايلات الحواف التشريحية لحالتي `ITF` و`RDC` عبر $p \in \{1.5, 2.0, 2.5\}$ |
| **`Stage 8`** | التحليل الإشعاعي السريري (`ROI Contrast-to-Noise & Modulation Transfer Function`) | ⏳ قادمة | تقييم الحفاظ على التباين والضجيج في الرئة، الكبد، الأبهر، والعمود الفقري |
| **`Stage 9`** | التجميع النهائي للأشكال والجداول عالية الدقة الجاهزة لمذكرة الماستر وأرشفة قابلية التكرار الكاملة | ⏳ قادمة | حزمة نتائج وأشكال علمية موثّقة بالكامل وجاهزة للمناقشة الأكاديمية |

---

## 1. جدول القرارات العلمية والهندسية المعتمدة (المرجع الدستوري للمشروع)

| المحور | القرار المعتمد | المبرر الفيزيائي والهندسي |
| :--- | :--- | :--- |
| **هوية البحث** | إعادة تنفيذ علمية مستقلة وموثّقة بالكامل مستلهمة من ملصق AAPM 2026 دون الاعتماد على مراسلة المؤلفين | ضمان الاستقلالية العلمية وتوثيق كل فرضية ومعادلة بدقة تصلح للمناقشة الأكاديمية |
| **منهجية العمل** | نظام **Stage-Gate** صارم: كود + توثيق تفصيلي مملّ + اختبارات متعددة المستويات لكل مرحلة قبل الانتقال | منع تراكم الأخطاء الهندسية وعدم الانتقال لأي مرحلة تالية قبل اجتياز بوابة القبول |
| **بوابة المقارنة البصرية الإلزامية** | توليد تلقائي للوحات مقارنة بصرية (`PNG` جنبًا إلى جنب مع الأصل/المرجع التحليلي وخرائط الخطأ) في كل مرحلة قابلة للمعاينة البصرية | الأرقام وحدها قد تخفي انقلابًا مكانيًا أو تشوهًا هندسيًا موضعيًا؛ الفحص البصري المباشر مع الأصل شرط إلزامي في كل مرحلة |
| **إدارة الـ Backends والمدخلات** | تحديد صريح للـ Backend (`numpy`, `torch`, `torch_cuda`) **ومنْع أي انتقال صامت (`No silent fallback`)** أو تخطي صامت لمريض غير موجود | ضمان قابلية تكرار التجارب وكشف أي خطأ في المسارات أو غياب الـ GPU فورًا |
| **الوعي الذكي بالبيئة** | اكتشاف تلقائي للبيئة (`kaggle` مقابل `full`): على Kaggle يلتزم بسقف **$\le 18.0\text{ GiB}$** في `/kaggle/working` ويضع الـ `.zip` المؤقت في `/tmp` ويحذفه بعد الاستخراج؛ وخارج Kaggle ينزّل كامل القائمة | تمكين التجريب الآمن والسريع لـ $10\text{–}12$ مريضًا على Kaggle ($2\times\text{T4}$)، ثم تشغيل الكود نفسه دون تعديل على خادم أكبر لكامل البيانات |
| **قراءة DICOM-CT-PD** | فكّ ثنائي صريح للوسوم الخاصة (`VR=UN`) وفق دليل `DICOM-CT-PD v4`، ترتيب وفحص استمرارية الزاوية و$z_0$، وتحويل المصفوفة بـ `pixel_array.T` (`order='F'`) إلى `(64, 736)` دون تكرار اللوغاريتم (`LogFlag=YES`) | ملفات TCIA مكتوبة بـ `Implicit VR Little Endian` وبترتيب أعمدة MATLAB (`Rows=736, Columns=64`)؛ أي قراءة بـ `order='C'` تخلط الصفوف بالأعمدة |
| **تقدير الـ Pitch والهندسة** | حساب ديناميكي لكل سلسلة عبر انحدار خطي لـ $z_0(\phi_{\text{unwrap}})$ لاستخراج `table_feed_mm_per_rot` و`normalized_pitch` واتجاهي الدوران والطاولة | عدم افتراض ثبات الـ pitch عند `0.9` أو `0.6` أو افتراض إشارة دوران موجبة دائمًا (في `C012` تتناقص $\phi_0$ بمقدار $-\frac{2\pi}{1152}$ لكل view) |
| **البؤرة المتحركة (`FFS`)** | قراءة وتطبيق المركبات الثلاث $(\rho_0+\Delta\rho,\; \phi_0+\Delta\phi,\; z_0+\Delta z)$ لكل view في كلٍّ من `Curved-to-Flat` و`SSRB` و`forward_project_helical_3d` | في جهاز `SOMATOM Definition AS+` حتى عند `FFSZ` تنزاح البؤرة شعاعيًا بـ $\Delta\rho = +5.45\text{ mm}$ بسبب ميل المصعد ($\approx 6.9^\circ$)، وإهمالها يزيح الأوتار بـ $1.64\text{ mm}$ |
| **الربط بين الصور والإسقاطات** | استخدام تسلسل `ReferenceRawDataSequence (0008,9121)` $\rightarrow$ `(0008,1115)` $\rightarrow$ `(0020,000E)` | في بيانات TCIA يختلف كلٌّ من `StudyInstanceUID` و`FrameOfReferenceUID` بين الصور المنشورة والإسقاطات الخام |
| **Curved-to-Flat & SSRB** | تقاطع شعاع ثلاثي الأبعاد دقيق مع السطح الأسطواني/المسطح/الكروي، ومطابقة وتر الشعاع الأفقي تحت `3D FFS`، وقناع `measured_mask` صريح يحظر `mode='nearest'`، واختيار $\arg\min \|v^\star\|$ | تصحيح الأخطاء الأربعة في `Helix2Fan` وتمثيل نقص البيانات الفعلي الناتج عن الـ high-pitch بأمانة تامة |
| **المرجع (`Reference Reconstruction`)** | إعادة بناء $x_{\text{reference}}$ من الإسقاطات الخام كاملة الجرعة عند الـ pitch الأصلي لكل فحص عبر خط `SSRB + Exact Parker + Spatial Ram-Lak FBP` (`ram_lak` مرشح رئيسي) مع مطابقة `FOV` الشركة (`400.0 mm`) | الحفاظ على أعلى دقة مكانية (`Ram-Lak`) ومطابقة المقياس الفيزيائي `1:1` مع صور الشركة دون تعلم تقليد مرشح تجاري خاص |
| **محاكاة `RDC` و`ITF` وإعادة الإسقاط** | تنفيذ السيناريوهين `RDC` (تقليص الصفوف النشطة `Fig. 1(b)`) و`ITF` (زيادة سرعة الطاولة عبر مسقط حلزوني ثلاثي الأبعاد `Fig. 1(c)`) وتوليد $x_{\text{prelim}}$ و$S_{\text{reproj}}$ (`Fig. 2(a)-(d)`) | مطابقة الفيزياء الفعلية لمسار الطاولة وتطبيق البنية الرياضية الموصوفة في ملصق AAPM 2026 |
| **التدريب ثنائي المجال (`Stage 5`)** | مزيج فيزيائي هجين **`60% ITF + 40% RDC`** عبر الخطوات السريرية **`p ∈ {1.5, 2.0, 2.5}`** مع التكيّف التلقائي لـ `native_pitch` المريض، وطبقة `HardDataConsistencyLayer`، وجسر `DifferentiableParkerFBPBridge` بمرشح `ram_lak` | الجمع بين الفيزياء السريرية الحقيقية لحركة الطاولة (`60% ITF`) والضجيج الفوتوني الحقيقي (`40% RDC`) ضمن النطاق السريري الفعلي وتجنّب القيم المتطرفة غير الواقعية |

---

## 2. تفاصيل المراحل المنفذة (`Stage 0` $\rightarrow$ `Stage 5`)

### أ. المرحلة 0 (`Stage 0 — v0.3.1`): التنزيل المتوازي الذكي والفهرسة السريعة
- راجع [`docs/01_data_download.md`](./docs/01_data_download.md) و[`docs/02_data_organization.md`](./docs/02_data_organization.md): كشف تلقائي لبيئة `kaggle` (`<= 18.0 GiB` وتخزين `.zip` في `/tmp`)، جلسات `requests.Session` مجمّعة بكتل `4 MiB` مع `SHA-256` لحظي واستخراج ذري بتمريرة واحدة مع فحص `CRC-32`، وفهرسة رؤوس `DICOM-CT-PD v4` المتوازية.

### ب. المرحلة 1 (`Stage 1 — v0.4.0 / v0.5.0`): الهندسة الحلزونية ثلاثية الأبعاد، القارئ الآمن للذاكرة، البراهين الفيزيائية للتوجيه، والفانتوم التحليلي
- راجع [`docs/04_stage1_reader_and_phantom.md`](./docs/04_stage1_reader_and_phantom.md):
  - **`geometry.py` (`HelicalGeometry`):** نموذج ثلاثي الأبعاد صريح للنظام الأيسر وفصل $\vec{F}_0(i)$ عن البؤرة الفعلية $\vec{S}(i)$ مع `3D FFS` وتحويل `0-based` لمركز الكاشف.
  - **`ctpd_reader.py`:** قراءة `order='F'` المطابقة لـ `pixel_array.T` دون تكرار اللوغاريتم، تدفق خفيف الذاكرة، والبراهين الفيزيائية الكمية الثلاثة (`verify_projection_orientation`) المحققة على المريض الحقيقي `C012` على `Kaggle`.
  - **`phantom.py`:** فانتوم الصدر ثلاثي الأبعاد غير المتناظر والمسقط الحلزوني التحليلي المغلق ومحوّل الشرائح الإشعاعي `voxelize_phantom_slices_radiological`.

### ج. المرحلة 2 (`Stage 2 — v0.5.0 / v0.6.0`): التحويل المتجهي والتدفقي `Curved-to-Flat` + `Helical SSRB` مع `3D FFS` و`measured_mask`
- راجع [`docs/05_stage2_rebinning_ssrb.md`](./docs/05_stage2_rebinning_ssrb.md):
  - **`rebin_curved_to_flat` و`rebin_helical_ssrb`:** تقاطع ثلاثي الأبعاد دقيق مع تعويض كامل للبؤرة المتحركة $(\Delta\rho, \Delta\phi, \Delta z)$، تطابق حتمي عبر معالجات `Intel Xeon AVX-512`، وتدفق محدود الذاكرة `rebin_series_ssrb_streaming`.
  - **مُثبت فعليًا على `C012` في `Kaggle`:** اكتمال قياس **`100.00%`** في `short_scan` عبر جميع شرائح الصدر الخمس عند `pitch = 0.8966`، مقابل **`84.10%`** (أي `15.90%` فجوات حقيقية مصفّرة بصدق) عند محاكاة `High-Pitch RDC` (`eff. pitch = 1.7932`).

### د. المرحلة 3 (`Stage 3 — v0.6.0 / v0.7.0`): استكمال الجوانب (`Sinogram Completion`) + ترجيح `Parker` الدقيق + إعادة البناء المرجعية `Ram-Lak Fan-Beam FBP` ($x_{\text{reference}}$)
- راجع التوثيق الرياضي والفيزيائي المملّ في [`docs/06_stage3_parker_fbp_reference.md`](./docs/06_stage3_parker_fbp_reference.md):
  1. **`extrapolate_sinogram` (`conjugate_feather`):** استعادة الفجوات المفقودة عند الـ `High-Pitch` أولًا من أشعتها المرافقة المقاسة فعليًا $(\beta \pm (\pi + 2\alpha_m), -u_m)$ متبوعةً بدمج زاوي وجانبي سلس وتمديد جانبي بجيب التمام المرفوع (`lateral_pad_cols=96`) لمنع تقعّر الأطراف، مع **عدم المساس إطلاقًا بقناع `SSRBResult.measured_mask` الأصلي**.
  2. **`compute_parker_weights`:** أوزان `Parker (1982)` الملساء الدقيقة بـ $\sin^2$ مع حلّ الركنين الشاذين `0/0` بـ `w = 0.5` لتحقيق وحدة تجزئة المرافق $w(\tilde{\beta}, \alpha) + w(\tilde{\beta}+\pi+2\alpha, -\alpha) \equiv 1$ وتكامل العمود $\sum w_k\Delta\beta = \pi$ بدقة `< 1e-13`.
  3. **`reconstruct_fanbeam_fbp` (`ram_lak` مرشح رئيسي افتراضي):** معادلة `Kak & Slaney` الدقيقة للمروحة متساوية التباعد بالاستجابة النبضية المكانية لمرشح `Ram-Lak` (`k[0] = 1/(4*du_iso)`) التي تعطي معايرة ماء صفرية الانحياز (**`0.0 HU`** دون تقعّر)، بتوجيه سريري قياسي (`Supine Radiological Grid`) ومطابقة تلقائية لمجال رؤية الشركة (`400.0 mm`).

### هـ. المرحلة 4 (`Stage 4 — v0.7.0`): محرك الإسقاط الأمامي (`2D Fan-Beam` + `3D Helical Cone-Beam`) ومحاكاة `RDC` و`ITF` وإعادة الإسقاط $S_{\text{reproj}}$ (`Fig. 2(a)-(d)`)
- راجع التوثيق الرياضي والفيزيائي المملّ في [`docs/07_stage4_forward_projection_and_simulation.md`](./docs/07_stage4_forward_projection_and_simulation.md):
  1. **`forward_project_fanbeam_2d` و`forward_project_fanbeam_2d_scalar_ref`:** مسقط شعاعي مروحي ثنائي الأبعاد متوافق تمامًا مع جملة إحداثيات `reconstruct_fanbeam_fbp` (`Ram-Lak` Roundtrip `SSIM >= 0.90` ومعايرة ماء `0.0 HU` ضمن `±1.5 HU`).
  2. **`compute_rdc_active_row_range` (`Scenario 1: RDC — Fig. 1(b)`):** تحديد الصفوف المركزية النشطة تلقائيًا بدلالة $p_{\text{native}}$ و$p_{\text{target}}$.
  3. **`synthesize_itf_helical_geometry` و`forward_project_helical_3d` (`Scenario 2: ITF — Fig. 1(c)`):** مضاعفة سرعة الطاولة مع قفل الطور الزاوي عند مركز الشريحة $z_{\text{mid}}$ والحفاظ على التذبذب الدوري الحقيقي للبؤرة الطائرة $\Delta z_{\text{FFS}}$ وإسقاط الحجم ثلاثي الأبعاد عبر جميع صفوف الكاشف الـ $64$.
  4. **`build_dual_domain_pair_from_ssrb` و`simulate_series_high_pitch_streaming`:** توليد جميع مصفوفات الشكل `2(a)-(d)` (`s_target`, `s_hp_raw`, `mask_hp`, `s_hp_extrap`, `s_hp_parker_weighted`, `x_prelim_hu`, `s_reproj`, `s_reproj_dc`) ولوحات المقارنة البصرية ذات الـ 8 نوافذ.

### و. المرحلة 5 (`Stage 5 — v0.8.0`): التعلّم العميق ثنائي المجال المشروط فيزيائيًا (`Dual-Domain Deep Learning`)
- راجع التوثيق الرياضي والفيزيائي في [`docs/08_stage5_dual_domain_dl.md`](./docs/08_stage5_dual_domain_dl.md):
  1. **`SinoInpaintUNet` + `HardDataConsistencyLayer`:** شبكة إكمال الساينوغرام بـ 4 قنوات (`[s_hp_raw, s_reproj, mask_hp, pitch_wedge_cond]`) متبوعة بطبقة اتساق فيزيائي صارم تحفظ الأشعة المقاسة بنسبة `100%`.
  2. **`DifferentiableParkerFBPBridge`:** جسر `Parker + Cosine + Ram-Lak FFT + Fan-Beam Backprojection` تفاضلي في `PyTorch` مطابق لـ `reconstruct_fanbeam_fbp` بدقة `< 0.05 HU`.
  3. **`ImageRefineUNet`:** شبكة تنقية الصورة المقطعية بـ 3 قنوات (`[x_dc_norm, x_prelim_norm, fov_pitch_map]`) لإزالة خطوط الـ `High-Pitch` المتبقية داخل `fov_mask`.
  4. **`build_weighted_hybrid_curriculum` (`60% ITF + 40% RDC`, `p ∈ {1.5, 2.0, 2.5}`):** تدريب وتقييم هجين متوازن عبر الخطوات السريرية الثلاث مع لوحات مقارنة بصرية سداسية (`stage5_*_dl_visual_qa.png`).

---

## 3. بنية المستودع والمخرجات

```text
HighPitchCT-Pipeline/
├── configs/
│   ├── pilot_c012.csv                          # قائمة موثّقة للمريض C012 (13,446 إسقاطًا كامل الجرعة + 351 صورة منشورة)
│   └── series.example.csv                      # قالب توضيحي لملف السلاسل
├── docs/
│   ├── 00_project_overview.md                  # النظرة العامة وخارطة المراحل التسع
│   ├── 01_data_download.md                     # توثيق تفصيلي للتنزيل فائق السرعة وحارس الـ 18 GiB وشريط التقدم
│   ├── 02_data_organization.md                 # توثيق تفصيلي لفك وسوم DICOM-CT-PD v4 وحساب الـ Pitch والتدقيق
│   ├── 03_kaggle.md                            # دليل تشغيل Kaggle (T4 x2 + 30GB RAM + سقف 18 GiB + Kaggle Secrets)
│   ├── 04_stage1_reader_and_phantom.md         # التوثيق الرياضي والفيزيائي الكامل للمرحلة 1 (الهندسة والقارئ والفانتوم)
│   ├── 05_stage2_rebinning_ssrb.md             # التوثيق الرياضي والفيزيائي الكامل للمرحلة 2 (Curved-to-Flat + SSRB + Visual QA)
│   ├── 06_stage3_parker_fbp_reference.md       # التوثيق الرياضي والفيزيائي الكامل للمرحلة 3 (Parker + Spatial Ram-Lak FBP + x_reference)
│   ├── 07_stage4_forward_projection_and_simulation.md # التوثيق الرياضي والفيزيائي الكامل للمرحلة 4 (Forward Projector + RDC & ITF + Fig. 2(d))
│   ├── 08_stage5_dual_domain_dl.md             # التوثيق الرياضي والفيزيائي الكامل للمرحلة 5 (Dual-Domain DL + 60/40 ITF/RDC + p=1.5,2.0,2.5)
│   ├── validation_protocol.md                  # بروتوكول التحقق وبوابة المقارنة البصرية الإلزامية
│   └── decisions/                              # سجلات القرارات المعمارية (ADR-0001 إلى ADR-0004)
├── notebooks/
│   ├── 00_kaggle_stage0.ipynb                  # دفتر Kaggle للمرحلة 0 (التنزيل والفهرسة)
│   ├── 01_kaggle_stage1.ipynb                  # دفتر Kaggle للمرحلة 1 (التدقيق الفيزيائي ومعاينة C012 والفانتوم)
│   ├── 02_kaggle_stage2.ipynb                  # دفتر Kaggle للمرحلة 2 (SSRB + عرض لوحات المقارنة البصرية الإلزامية)
│   ├── 03_kaggle_stage3.ipynb                  # دفتر Kaggle للمرحلة 3 (Parker Ram-Lak FBP + FOV=400mm + لوحات المقارنة البصرية)
│   ├── 04_kaggle_stage4.ipynb                  # دفتر Kaggle للمرحلة 4 (الإسقاط الأمامي + محاكاة RDC و ITF + Fig. 2(a)-(d))
│   └── 05_kaggle_stage5.ipynb                  # دفتر Kaggle للمرحلة 5 (التدريب والتقييم ثنائي المجال 60% ITF + 40% RDC عبر p=1.5,2.0,2.5)
├── src/highpitchct/
│   ├── cli.py                                  # واجهة الأوامر الموحدة hpct (شاملة stage1-audit -> stage5-train)
│   ├── ctpd_reader.py                          # قارئ DICOM-CT-PD الآمن للذاكرة ومحرك براهين التوجيه الثلاثة
│   ├── ctpd_tags.py                            # مفكك الوسوم الثنائية الخاصة وفق DICOM-CT-PD v4 ورابط (0008,9121)
│   ├── dual_domain_dl.py                       # المرحلة 5: الشبكة ثنائية المجال والجسر التفاضلي والتدريب الهجين 60/40 ITF/RDC
│   ├── forward_project.py                      # المرحلة 4: المسقط الأمامي 2D/3D ومحاكاة RDC و ITF وتوليد أزواج Fig. 2(a)-(d)
│   ├── geometry.py                             # نموذج الهندسة الحلزونية ثلاثي الأبعاد (HelicalGeometry + 3D FFS)
│   ├── index_dicom.py                          # الفهرسة المتوازية وتدقيق اكتمال السلاسل وتقدير الـ Pitch
│   ├── layout.py                               # مسارات التخزين الحتمية وفصل incoming المؤقت عن raw الدائم
│   ├── manifest.py                             # قراءة وكتابة والتحقق من قوائم CSV للسلاسل
│   ├── phantom.py                              # فانتوم الصدر ثلاثي الأبعاد والإسقاط التحليلي ومحوّل الشرائح الإشعاعي
│   ├── progress.py                             # محرّك أشرطة التقدم الحيّة الآمن للخيوط (tqdm + بديل قياسي)
│   ├── rebinning.py                            # المرحلة 2: Curved-to-Flat و Helical SSRB المتجهي والتدفقي مع 3D FFS
│   ├── recon_fbp.py                            # المرحلة 3: استكمال السينوغرام وترجيح Parker الدقيق و Ram-Lak Fan-Beam FBP (x_reference)
│   ├── runtime.py                              # كاشف البيئة الذكي (kaggle / full) وحارس الميزانية DiskBudgetGuard
│   ├── safe_zip.py                             # استخراج ذري بتمريرة واحدة مع فحص CRC-32 وحماية Zip-Slip
│   ├── tcia_client.py                          # عميل NBIA المتوازي عالي الإنتاجية مع البصمة اللحظية وتقارير السرعة
│   └── visual_qa.py                            # مولّد لوحات المقارنة البصرية الإلزامية بصيغة PNG لجميع المراحل
└── tests/                                      # حزمة الاختبارات الشاملة (52 اختبارًا تغطي Stage 0 -> Stage 5 بصفر تحذيرات)
```

---

## 4. أوامر التشغيل والتحديث المباشرة على Kaggle (`Stage 4 + Stage 5 — v0.8.0`)

شغّل الخلايا التالية في دفتر Kaggle (أو افتح الدفتر الجاهز [`notebooks/05_kaggle_stage5.ipynb`](./notebooks/05_kaggle_stage5.ipynb)) لتحديث المستودع إلى `v0.8.0`، التحقق من نجاح جميع الاختبارات الـ **`52/52` بصفر تحذيرات**، تشغيل `Stage 4` و`Stage 5` بالبروتوكول السريري المعتمد (`60% ITF + 40% RDC`، `p ∈ {1.5, 2.0, 2.5}`، `ram_lak`، و`FOV = 400.0 mm`) وعرض جميع لوحات المقارنة البصرية مباشرةً:

```python
import os
from pathlib import Path
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_dir = Path("/kaggle/working/HighPitchCT-Pipeline")
if not repo_dir.exists():
    os.system(f"git clone https://{token}@github.com/BrahimSoufghalem/HighPitchCT-Pipeline.git {repo_dir}")
else:
    os.system(f"git -C {repo_dir} pull --ff-only")

%cd /kaggle/working/HighPitchCT-Pipeline
!python -m pip install -q -e .
!python -W error::DeprecationWarning -m unittest discover -s tests -t . -v
```

```python
import torch

sim_backend = "torch_cuda" if torch.cuda.is_available() else "numpy"
dl_backend = "torch_cuda" if torch.cuda.is_available() else "torch"
print("Selected explicit backends -> sim:", sim_backend, "| dl:", dl_backend)

!hpct download --series-file configs/pilot_c012.csv --workers 4 --retries 5
!hpct stage4-sim --case-id C012 --scenario both --target-pitch 2.0 --pitches 1.5,2.0,2.5 --num-slices 1 --n-xy 256 --itf-slab-slices 11 --filter-name ram_lak --backend {sim_backend} --run-phantom-check --save-pairs-npz --workers 4
!hpct stage5-train --case-id C012 --scenario both --pitches 1.5,2.0,2.5 --itf-ratio 0.60 --num-slices 1 --image-size 256 --itf-slab-slices 7 --epochs 12 --lr 0.0025 --filter-name ram_lak --backend {dl_backend} --run-phantom-check --save-dataset-npz --save-model --workers 4
```

---

## 5. بوابات القبول المنجزة

- [x] **بوابة `Stage 0` (`v0.3.1` — مُثبتة على Kaggle):** تنزيل متوازٍ بحارس `<= 18.0 GiB` وفهرسة `C012` (`13,446` مسقطًا + `351` صورة، `VALID`).
- [x] **بوابة `Stage 1` (`v0.5.0` — مُثبتة على `C012` في Kaggle):** [`HelicalGeometry`](./src/highpitchct/geometry.py)، [`ctpd_reader.py`](./src/highpitchct/ctpd_reader.py)، البراهين الفيزيائية الثلاثة، وفانتوم الصدر ثلاثي الأبعاد [`phantom.py`](./src/highpitchct/phantom.py).
- [x] **بوابة `Stage 2` (`v0.5.0` / `v0.6.0` — مُثبتة على `C012` في Kaggle):** [`rebinning.py`](./src/highpitchct/rebinning.py) (`Curved-to-Flat` + `SSRB` مع `3D FFS` وقناع `measured_mask` الصريح وحتمية حدود `Xeon AVX-512`).
- [x] **بوابة `Stage 3` (`v0.6.0` / `v0.7.0` — مُثبتة على `C012` في Kaggle):** [`recon_fbp.py`](./src/highpitchct/recon_fbp.py) (استكمال الأشعة المرافقة دون لمس `measured_mask`، ترجيح `Parker` الدقيق، `Spatial Ram-Lak FBP` كمرشح رئيسي بمعايرة ماء `0.0 HU`، ومطابقة `FOV = 400.0 mm` مع صور `Siemens`).
- [x] **بوابة `Stage 4` (`v0.7.0` — مُثبتة على `C012` في Kaggle):** [`forward_project.py`](./src/highpitchct/forward_project.py) (محرك الإسقاط الأمامي `2D Fan-Beam` و`3D Helical Cone-Beam`، محاكاة `RDC` و`ITF`، توليد أزواج `Fig. 2(a)-(d)` كاملةً شاملةً $S_{\text{reproj}}$، ولوحات المقارنة البصرية الثمانية).
- [x] **بوابة `Stage 5` (`v0.8.0` — مكتملة بـ `52/52` اختبارًا):** [`dual_domain_dl.py`](./src/highpitchct/dual_domain_dl.py) (الشبكة ثنائية المجال `SinoInpaintUNet` + `HardDataConsistencyLayer` + الجسر الفيزيائي التفاضلي `DifferentiableParkerFBPBridge` + `ImageRefineUNet`، والتدريب الهجين `60% ITF + 40% RDC` عبر `p ∈ {1.5, 2.0, 2.5}` مع التكيّف التلقائي لـ `native_pitch` المريض ولوحات المقارنة البصرية السداسية).

راجع الشرح التفصيلي المملّ لكل خوارزمية ومعادلة وبنية ملفات في مجلد [`docs/`](./docs/).
