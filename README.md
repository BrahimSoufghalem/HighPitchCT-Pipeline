# HighPitchCT-Pipeline

إطار بحثي وهندسي محكم وقابل للتدقيق لإعادة بناء وتقييم نموذج **ثنائي المجال (Dual-Domain Deep Learning)** لإعادة بناء التصوير المقطعي الحلزوني مرتفع الـ Pitch (**High-Pitch Helical CT**) انطلاقًا من بيانات الإسقاط الخام في مجموعة **TCIA (`LDCT-and-Projection-data`)**، ومُصمَّم خصيصًا لمذكرة ماستر في **الفيزياء الطبية** وفق نظام بوابات المراحل الصارم (**Stage-Gate**) وبوابة **المقارنة البصرية الإلزامية (`Mandatory Visual QA`)**.

> **الإصدار الحالي:** `v0.6.0` — **المراحل `Stage 0` و`Stage 1` و`Stage 2` مُثبتة فعليًا على المريض الحقيقي `C012` على Kaggle، والمرحلة `Stage 3` مكتملة ومحققة بـ `42/42` اختبارًا**: استكمال الفجوات الفيزيائي بالأشعة المرافقة (`extrapolate_sinogram`) مع صون `measured_mask` دون أي تعديل، ترجيح `Parker (1982)` الدقيق للمسح القصير مع حلّ الركنين الشاذين `0/0`، إعادة البناء المرجعية `Equispaced Fan-Beam FBP` بالاستجابة المكانية الدقيقة لمرشح `Ram-Lak/Hann` (`0.0 HU` معايرة ماء صفرية الانحياز) على `numpy` و`torch_cuda` دون أي انتقال صامت، وإنتاج المرجع الموحد **$x_{\text{reference}}$** ولوحات المقارنة البصرية ذات الـ 6 نوافذ (`stage3_*_fbp_visual_qa.png`)، بالإضافة إلى إصلاح حتمية الحدود `1-ULP` لمعالجات `Intel Xeon AVX-512` في `SSRB`.

---

## 0. خارطة المراحل الكاملة للمشروع (`Stage 0` $\rightarrow$ `Stage 9`)

| المرحلة | الاسم والمحتوى العلمي والهندسي | الحالة | المخرجات وبوابة القبول |
| :--- | :--- | :---: | :--- |
| **`Stage 0`** | اكتشاف وتنزيل `TCIA` فائق السرعة الواعي ببيئة `Kaggle <= 18 GiB` + فهرسة رؤوس `DICOM-CT-PD v4` | ✅ **مكتملة ومُثبتة على Kaggle (`v0.3.1`)** | `13,446` إسقاطًا + `351` صورة للمريض `C012` بحالة `VALID` (`1.419 GiB`، `pitch = 0.8966`) |
| **`Stage 1`** | الهندسة الحلزونية ثلاثية الأبعاد (`3D FFS`) + قارئ `DICOM-CT-PD` الآمن للذاكرة + 3 براهين توجيه فيزيائية + فانتوم الصدر ثلاثي الأبعاد + `Visual QA` | ✅ **مكتملة ومُثبتة على `C012` في Kaggle (`v0.5.0`)** | `geometry.py`, `ctpd_reader.py`, `phantom.py`, اجتياز البراهين الثلاثة على `C012` في `49.5s` ومعاينة الصور البصرية |
| **`Stage 2`** | إعادة الترتيب الهندسي المتجهي والتدفقي: `Curved-to-Flat` + `SSRB` مع معالجة `3D FFS` وقناع القياس الصريح `measured_mask` ولوحات المقارنة البصرية | ✅ **مكتملة ومُثبتة على `C012` في Kaggle (`v0.5.0` / `v0.6.0`)** | تغطية `100.00%` في `Short-Scan` لـ 5 شرائح صدر من `C012` عند `pitch=0.90`، وتغطية `84.10%` عند `High-Pitch RDC (eff. pitch=1.79)` + إصلاح حدود `Xeon AVX-512` |
| **`Stage 3`** | استكمال الجوانب (`Conjugate-Ray + Feathering`) + ترجيح `Parker` الدقيق للمسح القصير + `Spatial Ram-Lak/Hann FBP` وإنتاج المرجع $x_{\text{reference}}$ + مقارنات صور `FBP` | ✅ **مكتملة برمجيًا (`v0.6.0`) — جاهزة لتشغيل `C012` على Kaggle** | `recon_fbp.py`, `hpct stage3-recon`, معايرة ماء `0.0 HU`، `SSIM = 0.947` على الفانتوم (`pitch=0.85`)، دعم صريح لـ `torch_cuda`، ولوحات مقارنة بصرية ذات 6 نوافذ |
| **`Stage 4`** | محرك الإسقاط الأمامي الحلزوني (`LEAP` / مسقط ثلاثي الأبعاد صريح على GPU) ومحاكاة سيناريوهات `High-Pitch` (`ITF` و`RDC`) وإعادة الإسقاط $p_{\text{in}}$ | ⏭ **التالية مباشرةً** | توليد أزواج تدريب فيزيائية متطابقة مع ملصق `AAPM 2026` وفحص الثبات الذاتي `Forward -> FBP` |
| **`Stage 5`** | بناء مجموعة البيانات وتقسيم المرضى الصارم (`Patient-level Split`) لبيئتي `Kaggle` والخادم الكامل | ⏳ قادمة | تغليف عينات التدريب والتحقق والاختبار مع منع تسرب الشرائح بين المرضى (`Zero Leakage`) |
| **`Stage 6`** | طبقة `Differentiable Short-Scan FBP` القابلة للاشتقاق في `PyTorch` (`autograd` على `2x T4`) | ⏳ قادمة | اجتياز فحص التدرجات العددية `torch.autograd.gradcheck` وتطابق الخرج مع `Stage 3` |
| **`Stage 7`** | الشبكة العصبية ثنائية المجال (`Sinogram Residual U-Net` + `Differentiable FBP` + `Image Residual U-Net`) وخسارة $\mathcal{L}_1 + \mathcal{L}_\nabla$ | ⏳ قادمة | تدريب مستقر بـ `AMP / DataParallel` واجتياز اختبار `Overfit` على حالة واحدة |
| **`Stage 8`** | مصفوفة التقييم الكمي الشامل ودراسات الحذف (`Ablations`: `FBP` مقابل `Sino-only` مقابل `Image-only` مقابل `Dual-Domain`) | ⏳ قادمة | جداول ومنحنيات `RMSE [HU]`, `PSNR`, `SSIM`, وبروفايلات الحواف التشريحية لحالتي `ITF` و`RDC` |
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
| **البؤرة المتحركة (`FFS`)** | قراءة وتطبيق المركبات الثلاث $(\rho_0+\Delta\rho,\; \phi_0+\Delta\phi,\; z_0+\Delta z)$ لكل view في كلٍّ من `Curved-to-Flat` و`SSRB` | في جهاز `SOMATOM Definition AS+` حتى عند `FFSZ` تنزاح البؤرة شعاعيًا بـ $\Delta\rho = +5.45\text{ mm}$ بسبب ميل المصعد ($\approx 6.9^\circ$)، وإهمالها يزيح الأوتار بـ $1.64\text{ mm}$ |
| **الربط بين الصور والإسقاطات** | استخدام تسلسل `ReferenceRawDataSequence (0008,9121)` $\rightarrow$ `(0008,1115)` $\rightarrow$ `(0020,000E)` | في بيانات TCIA يختلف كلٌّ من `StudyInstanceUID` و`FrameOfReferenceUID` بين الصور المنشورة والإسقاطات الخام |
| **Curved-to-Flat & SSRB** | تقاطع شعاع ثلاثي الأبعاد دقيق مع السطح الأسطواني/المسطح/الكروي، ومطابقة وتر الشعاع الأفقي تحت `3D FFS`، وقناع `measured_mask` صريح يحظر `mode='nearest'`، واختيار $\arg\min \|v^\star\|$ | تصحيح الأخطاء الأربعة في `Helix2Fan` وتمثيل نقص البيانات الفعلي الناتج عن الـ high-pitch بأمانة تامة |
| **المرجع (`Reference Reconstruction`)** | إعادة بناء $x_{\text{reference}}$ من الإسقاطات الخام كاملة الجرعة عند الـ pitch الأصلي لكل فحص عبر خط `SSRB + Exact Parker + Spatial Ram-Lak/Hann FBP` الخاص بنا | منع الشبكة من تعلم تقليد مرشح إعادة البناء التجاري (`B50f` مثلًا) وإزاحات جملة إحداثياته `LPS`، وحصر التعلم في إزالة تشوهات الـ high-pitch |
| **محاكاة ITF وRDC والتعلم** | تحويل $x_{\text{reference}}$ إلى $\mu(x,y,z)$ وتوليد إسقاطات أمامية حلزونية (`LEAP`) لحالات ITF/RDC، وتدريب شبكتي Residual U-Net مرتبطتين عبر `Differentiable FBP` بخسارة $\mathcal{L}_1 + \mathcal{L}_\nabla$ | مطابقة الفيزياء الفعلية لمسار الطاولة وتطبيق البنية الرياضية الموصوفة في ملصق AAPM 2026 |

---

## 2. تفاصيل المراحل المنفذة (`Stage 0` $\rightarrow$ `Stage 3`)

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

### د. المرحلة 3 (`Stage 3 — v0.6.0`): استكمال الجوانب (`Sinogram Completion`) + ترجيح `Parker` الدقيق + إعادة البناء المرجعية `Fan-Beam FBP` ($x_{\text{reference}}$)
- راجع التوثيق الرياضي والفيزيائي المملّ في [`docs/06_stage3_parker_fbp_reference.md`](./docs/06_stage3_parker_fbp_reference.md):
  1. **`extrapolate_sinogram` (`conjugate_feather`):** استعادة الفجوات المفقودة عند الـ `High-Pitch` أولًا من أشعتها المرافقة المقاسة فعليًا $(\beta \pm (\pi + 2\alpha_m), -u_m)$ متبوعةً بدمج زاوي وجانبي سلس وتمديد جانبي بجيب التمام المرفوع (`lateral_pad_cols=96`) لمنع تقعّر الأطراف، مع **عدم المساس إطلاقًا بقناع `SSRBResult.measured_mask` الأصلي**.
  2. **`compute_parker_weights`:** أوزان `Parker (1982)` الملساء الدقيقة بـ $\sin^2$ مع حلّ الركنين الشاذين `0/0` بـ `w = 0.5` لتحقيق وحدة تجزئة المرافق $w(\tilde{\beta}, \alpha) + w(\tilde{\beta}+\pi+2\alpha, -\alpha) \equiv 1$ وتكامل العمود $\sum w_k\Delta\beta = \pi$ بدقة `< 1e-13`.
  3. **`reconstruct_fanbeam_fbp` و`reconstruct_series_reference_streaming`:** معادلة `Kak & Slaney` الدقيقة للمروحة متساوية التباعد بالاستجابة النبضية المكانية لمرشح `Ram-Lak` (`k[0] = 1/(4*du_iso)`) التي تعطي معايرة ماء صفرية الانحياز (**`0.0 HU`** دون تقعّر)، بتوجيه سريري قياسي (`Supine Radiological Grid`) ودعم صريح لـ `numpy` و`torch` و`torch_cuda` دون أي انتقال صامت.
  4. **المقارنة البصرية الإلزامية (`save_stage3_phantom_fbp_visual_qa` و`save_stage3_real_fbp_visual_qa`):** لوحات من 6 نوافذ تقارن $x_{\text{reference}}$ بالأصل التحليلي وبصور `Siemens` المنشورة وبإعادة بناء الـ `High-Pitch` الخام والمستكملة.

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
│   ├── 06_stage3_parker_fbp_reference.md       # التوثيق الرياضي والفيزيائي الكامل للمرحلة 3 (Parker + Spatial FBP + x_reference)
│   ├── validation_protocol.md                  # بروتوكول التحقق وبوابة المقارنة البصرية الإلزامية
│   └── decisions/                              # سجلات القرارات المعمارية (ADR-0001 إلى ADR-0004)
├── notebooks/
│   ├── 00_kaggle_stage0.ipynb                  # دفتر Kaggle للمرحلة 0 (التنزيل والفهرسة)
│   ├── 01_kaggle_stage1.ipynb                  # دفتر Kaggle للمرحلة 1 (التدقيق الفيزيائي ومعاينة C012 والفانتوم)
│   ├── 02_kaggle_stage2.ipynb                  # دفتر Kaggle للمرحلة 2 (SSRB + عرض لوحات المقارنة البصرية الإلزامية)
│   └── 03_kaggle_stage3.ipynb                  # دفتر Kaggle للمرحلة 3 (Parker FBP + x_reference + لوحات المقارنة البصرية)
├── src/highpitchct/
│   ├── cli.py                                  # واجهة الأوامر الموحدة hpct (شاملة stage1-audit و stage2-rebin و stage3-recon)
│   ├── ctpd_reader.py                          # قارئ DICOM-CT-PD الآمن للذاكرة ومحرك براهين التوجيه الثلاثة
│   ├── ctpd_tags.py                            # مفكك الوسوم الثنائية الخاصة وفق DICOM-CT-PD v4 ورابط (0008,9121)
│   ├── geometry.py                             # نموذج الهندسة الحلزونية ثلاثي الأبعاد (HelicalGeometry + 3D FFS)
│   ├── index_dicom.py                          # الفهرسة المتوازية وتدقيق اكتمال السلاسل وتقدير الـ Pitch
│   ├── layout.py                               # مسارات التخزين الحتمية وفصل incoming المؤقت عن raw الدائم
│   ├── manifest.py                             # قراءة وكتابة والتحقق من قوائم CSV للسلاسل
│   ├── phantom.py                              # فانتوم الصدر ثلاثي الأبعاد والإسقاط التحليلي ومحوّل الشرائح الإشعاعي
│   ├── progress.py                             # محرّك أشرطة التقدم الحيّة الآمن للخيوط (tqdm + بديل قياسي)
│   ├── rebinning.py                            # المرحلة 2: Curved-to-Flat و Helical SSRB المتجهي والتدفقي مع 3D FFS
│   ├── recon_fbp.py                            # المرحلة 3: استكمال السينوغرام وترجيح Parker الدقيق و Fan-Beam FBP (x_reference)
│   ├── runtime.py                              # كاشف البيئة الذكي (kaggle / full) وحارس الميزانية DiskBudgetGuard
│   ├── safe_zip.py                             # استخراج ذري بتمريرة واحدة مع فحص CRC-32 وحماية Zip-Slip
│   ├── tcia_client.py                          # عميل NBIA المتوازي عالي الإنتاجية مع البصمة اللحظية وتقارير السرعة
│   └── visual_qa.py                            # مولّد لوحات المقارنة البصرية الإلزامية بصيغة PNG لجميع المراحل
└── tests/                                      # حزمة الاختبارات الشاملة (42 اختبارًا تغطي Stage 0 -> Stage 3 بصفر تحذيرات)
```

---

## 4. أوامر التشغيل والتحديث المباشرة على Kaggle (`Stage 3 — v0.6.0`)

شغّل الخلايا التالية في دفتر Kaggle (أو افتح الدفتر الجاهز [`notebooks/03_kaggle_stage3.ipynb`](./notebooks/03_kaggle_stage3.ipynb)) لتحديث المستودع إلى `v0.6.0`، التحقق من نجاح جميع الاختبارات الـ **`42/42` بصفر تحذيرات**، تشغيل إعادة البناء المرجعية `Stage 3` على المريض الحقيقي `C012` والفانتوم المرجعي (على `GPU T4` تلقائيًا إن كان مفعّلًا أو `numpy` على CPU)، وعرض لوحات المقارنة البصرية ذات الـ 6 نوافذ مباشرةً:

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

backend = "torch_cuda" if torch.cuda.is_available() else "numpy"
print("Selected explicit FBP backend:", backend)

!hpct download --series-file configs/pilot_c012.csv --workers 4 --retries 5
!hpct stage3-recon --case-id C012 --num-slices 5 --image-size 512 --filter hann --inpaint-mode conjugate_feather --backend {backend} --run-phantom-check --recon-out /kaggle/working/highpitchct-data/derived/C012_x_reference.npz --workers 8
```

```python
import json
from IPython.display import Image, Markdown, display

with open("/kaggle/working/highpitchct-data/manifests/stage3_recon_report.json") as f:
    r3 = json.load(f)

for title, path in [
    ("Stage 3 — مقارنة مرجع الفانتوم x_reference مع الأصل التحليلي و High-Pitch (1.85)", r3["synthetic_phantom_fbp_check"]["visual_qa_png"]),
    ("Stage 3 — المريض الحقيقي C012: شريحة Siemens مقابل مرجعنا x_reference مقابل High-Pitch RDC", r3["reconstructed_series"][0]["visual_qa_png"]),
]:
    display(Markdown(f"### {title}"))
    display(Image(filename=path))
```

---

## 5. بوابات القبول المنجزة

- [x] **بوابة `Stage 0` (`v0.3.1` — مُثبتة على Kaggle):** تنزيل متوازٍ بحارس `<= 18.0 GiB` وفهرسة `C012` (`13,446` مسقطًا + `351` صورة، `VALID`).
- [x] **بوابة `Stage 1` (`v0.5.0` — مُثبتة على `C012` في Kaggle):** [`HelicalGeometry`](./src/highpitchct/geometry.py)، [`ctpd_reader.py`](./src/highpitchct/ctpd_reader.py)، البراهين الفيزيائية الثلاثة، وفانتوم الصدر ثلاثي الأبعاد [`phantom.py`](./src/highpitchct/phantom.py).
- [x] **بوابة `Stage 2` (`v0.5.0` / `v0.6.0` — مُثبتة على `C012` في Kaggle):** [`rebinning.py`](./src/highpitchct/rebinning.py) (`Curved-to-Flat` + `SSRB` مع `3D FFS` وقناع `measured_mask` الصريح وحتمية حدود `Xeon AVX-512`).
- [x] **بوابة `Stage 3` (`v0.6.0` — مكتملة بـ `42/42` اختبارًا):** [`recon_fbp.py`](./src/highpitchct/recon_fbp.py) (استكمال الأشعة المرافقة دون لمس `measured_mask`، ترجيح `Parker` الدقيق، `Spatial Ram-Lak/Hann FBP` بمعايرة ماء `0.0 HU`، المرجع $x_{\text{reference}}$، ودعم `numpy` / `torch_cuda` الصريح + المقارنات البصرية السداسية).

راجع الشرح التفصيلي المملّ لكل خوارزمية ومعادلة وبنية ملفات في مجلد [`docs/`](./docs/).
