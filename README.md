# HighPitchCT-Pipeline

إطار بحثي وهندسي محكم وقابل للتدقيق لإعادة بناء وتقييم نموذج **ثنائي المجال (Dual-Domain Deep Learning)** لإعادة بناء التصوير المقطعي الحلزوني مرتفع الـ Pitch (**High-Pitch Helical CT**) انطلاقًا من بيانات الإسقاط الخام في مجموعة **TCIA (`LDCT-and-Projection-data`)**، ومُصمَّم خصيصًا لمذكرة ماستر في **الفيزياء الطبية** وفق نظام بوابات المراحل الصارم (**Stage-Gate**) وبوابة **المقارنة البصرية الإلزامية (`Mandatory Visual QA`)**.

> **الإصدار الحالي:** `v0.5.0` — **المراحل `Stage 0` و`Stage 1` و`Stage 2` مكتملة ومحققة بـ `36/36` اختبارًا**: اكتشاف وتنزيل `TCIA` بوعي ذكي لبيئة `Kaggle <= 18 GiB`، الهندسة الحلزونية ثلاثية الأبعاد (`HelicalGeometry` + `3D FFS`) وقارئ `DICOM-CT-PD v4` الآمن للذاكرة، التحويل الهندسي المتجهي السريع والتدفقي من كاشف منحنٍ إلى مسطح (`rebin_curved_to_flat`) وإعادة الترتيب الحلزوني أحادي الشريحة (`rebin_helical_ssrb` وفق Noo et al. 1999) مع تعويض كامل للبؤرة المتحركة ثلاثية الأبعاد $(\Delta\rho, \Delta\phi, \Delta z)$ وقناع القياس الصريح `measured_mask` دون أي تمديد زائف `mode='nearest'`، ونظام التوليد التلقائي للوحات المقارنة البصرية `PNG` (`visual_qa.py`).

---

## 0. خارطة المراحل الكاملة للمشروع (`Stage 0` $\rightarrow$ `Stage 9`)

| المرحلة | الاسم والمحتوى العلمي والهندسي | الحالة | المخرجات وبوابة القبول |
| :--- | :--- | :---: | :--- |
| **`Stage 0`** | اكتشاف وتنزيل `TCIA` فائق السرعة الواعي ببيئة `Kaggle <= 18 GiB` + فهرسة رؤوس `DICOM-CT-PD v4` | ✅ **مكتملة ومُثبتة على Kaggle (`v0.3.1`)** | `13,446` إسقاطًا + `351` صورة للمريض `C012` بحالة `VALID` (`pitch = 0.8966`) |
| **`Stage 1`** | الهندسة الحلزونية ثلاثية الأبعاد (`3D FFS`) + قارئ `DICOM-CT-PD` الآمن للذاكرة + 3 براهين توجيه فيزيائية + فانتوم الصدر ثلاثي الأبعاد + `Visual QA` | ✅ **مكتملة (`v0.5.0`)** | `geometry.py`, `ctpd_reader.py`, `phantom.py`, `hpct stage1-audit`, فانتوم مُثبت على Kaggle + مطابقة `C012 <-> case_C012` |
| **`Stage 2`** | إعادة الترتيب الهندسي المتجهي والتدفقي: `Curved-to-Flat` + `SSRB` مع معالجة `3D FFS` وقناع القياس الصريح `measured_mask` ولوحات المقارنة البصرية | ✅ **مكتملة برمجيًا (`v0.5.0`) — بانتظار تأكيد تشغيل `C012` على Kaggle** | `rebinning.py`, `visual_qa.py`, `hpct stage2-rebin`, `36/36` اختبارًا ناجحًا ومطابقة `Exact 2D GT` |
| **`Stage 3`** | استكمال الجوانب (`Extrapolation`) + ترجيح `Short-scan / Parker` + `FBP` القياسي وإنتاج المرجع $x_{\text{reference}}$ + مقارنات صور `FBP` مع الأصل | ⏭ **التالية مباشرةً** | إعادة بناء مرجعية نظيفة في جملة إحداثياتنا الموحدة وقياس `PSNR / SSIM / RMSE` ولوحات مقارنة بصرية مع الأصل |
| **`Stage 4`** | محرك الإسقاط الأمامي الحلزوني (`LEAP` صريح على GPU) ومحاكاة سيناريوهات `High-Pitch` (`ITF` و`RDC`) وإعادة الإسقاط $p_{\text{in}}$ | ⏳ قادمة | توليد أزواج تدريب فيزيائية متطابقة مع ملصق `AAPM 2026` وفحص الثبات الذاتي `Forward -> FBP` |
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
| **إدارة الـ Backends والمدخلات** | تحديد صريح للـ Backend في الإعدادات **ومنْع أي انتقال صامت (`No silent fallback`)** أو تخطي صامت لمريض غير موجود | ضمان قابلية تكرار التجارب وكشف أي خطأ في المسارات أو الإعدادات فورًا |
| **الوعي الذكي بالبيئة** | اكتشاف تلقائي للبيئة (`kaggle` مقابل `full`): على Kaggle يلتزم بسقف **$\le 18.0\text{ GiB}$** في `/kaggle/working` ويضع الـ `.zip` المؤقت في `/tmp` ويحذفه بعد الاستخراج؛ وخارج Kaggle ينزّل كامل القائمة | تمكين التجريب الآمن والسريع لـ $10\text{–}12$ مريضًا على Kaggle ($2\times\text{T4}$)، ثم تشغيل الكود نفسه دون تعديل على خادم أكبر لكامل البيانات |
| **قراءة DICOM-CT-PD** | فكّ ثنائي صريح للوسوم الخاصة (`VR=UN`) وفق دليل `DICOM-CT-PD v4`، ترتيب وفحص استمرارية الزاوية و$z_0$، وتحويل المصفوفة بـ `pixel_array.T` (`order='F'`) إلى `(64, 736)` دون تكرار اللوغاريتم (`LogFlag=YES`) | ملفات TCIA مكتوبة بـ `Implicit VR Little Endian` وبترتيب أعمدة MATLAB (`Rows=736, Columns=64`)؛ أي قراءة بـ `order='C'` تخلط الصفوف بالأعمدة |
| **تقدير الـ Pitch والهندسة** | حساب ديناميكي لكل سلسلة عبر انحدار خطي لـ $z_0(\phi_{\text{unwrap}})$ لاستخراج `table_feed_mm_per_rot` و`normalized_pitch` واتجاهي الدوران والطاولة | عدم افتراض ثبات الـ pitch عند `0.9` أو `0.6` أو افتراض إشارة دوران موجبة دائمًا (في `C012` تتناقص $\phi_0$ بمقدار $-\frac{2\pi}{1152}$ لكل view) |
| **البؤرة المتحركة (`FFS`)** | قراءة وتطبيق المركبات الثلاث $(\rho_0+\Delta\rho,\; \phi_0+\Delta\phi,\; z_0+\Delta z)$ لكل view في كلٍّ من `Curved-to-Flat` و`SSRB` | في جهاز `SOMATOM Definition AS+` حتى عند `FFSZ` تنزاح البؤرة شعاعيًا بـ $\Delta\rho = +5.45\text{ mm}$ بسبب ميل المصعد ($\approx 6.9^\circ$)، وإهمالها يزيح الأوتار بـ $1.64\text{ mm}$ |
| **الربط بين الصور والإسقاطات** | استخدام تسلسل `ReferenceRawDataSequence (0008,9121)` $\rightarrow$ `(0008,1115)` $\rightarrow$ `(0020,000E)` | في بيانات TCIA يختلف كلٌّ من `StudyInstanceUID` و`FrameOfReferenceUID` بين الصور المنشورة والإسقاطات الخام |
| **Curved-to-Flat & SSRB** | تقاطع شعاع ثلاثي الأبعاد دقيق مع السطح الأسطواني/المسطح/الكروي، ومطابقة وتر الشعاع الأفقي تحت `3D FFS`، وقناع `measured_mask` صريح يحظر `mode='nearest'`، واختيار $\arg\min \|v^\star\|$ | تصحيح الأخطاء الأربعة في `Helix2Fan` وتمثيل نقص البيانات الفعلي الناتج عن الـ high-pitch بأمانة تامة |
| **المرجع (`Reference Reconstruction`)** | إعادة بناء $x_{\text{reference}}$ من الإسقاطات الخام كاملة الجرعة عند الـ pitch الأصلي لكل فحص عبر خط `SSRB + Short-scan + Parker + FBP` الخاص بنا | منع الشبكة من تعلم تقليد مرشح إعادة البناء التجاري (`B50f` مثلًا) وإزاحات إحداثياته، وحصر التعلم في إزالة تشوهات الـ high-pitch |
| **محاكاة ITF وRDC والتعلم** | تحويل $x_{\text{reference}}$ إلى $\mu(x,y,z)$ وتوليد إسقاطات أمامية حلزونية (`LEAP`) لحالات ITF/RDC، وتدريب شبكتي Residual U-Net مرتبطتين عبر `Differentiable FBP` بخسارة $\mathcal{L}_1 + \mathcal{L}_\nabla$ | مطابقة الفيزياء الفعلية لمسار الطاولة وتطبيق البنية الرياضية الموصوفة في ملصق AAPM 2026 |

---

## 2. تفاصيل المراحل المنفذة (`Stage 0` + `Stage 1` + `Stage 2`)

### أ. المرحلة 0 (`Stage 0 — v0.3.1`): التنزيل المتوازي الذكي والفهرسة السريعة
- راجع [`docs/01_data_download.md`](./docs/01_data_download.md) و[`docs/02_data_organization.md`](./docs/02_data_organization.md): كشف تلقائي لبيئة `kaggle` (`<= 18.0 GiB` وتخزين `.zip` في `/tmp`)، جلسات `requests.Session` مجمّعة بكتل `4 MiB` مع `SHA-256` لحظي واستخراج ذري بتمريرة واحدة مع فحص `CRC-32`، وفهرسة رؤوس `DICOM-CT-PD v4` المتوازية.

### ب. المرحلة 1 (`Stage 1 — v0.4.0 / v0.5.0`): الهندسة الحلزونية ثلاثية الأبعاد، القارئ الآمن للذاكرة، البراهين الفيزيائية للتوجيه، والفانتوم التحليلي
- راجع [`docs/04_stage1_reader_and_phantom.md`](./docs/04_stage1_reader_and_phantom.md):
  - **`geometry.py` (`HelicalGeometry`):** نموذج ثلاثي الأبعاد صريح للنظام الأيسر وفصل $\vec{F}_0(i)$ عن البؤرة الفعلية $\vec{S}(i)$ مع `3D FFS` وتحويل `0-based` لمركز الكاشف.
  - **`ctpd_reader.py`:** قراءة `order='F'` المطابقة لـ `pixel_array.T` دون تكرار اللوغاريتم، تدفق خفيف الذاكرة، والبراهين الفيزيائية الكمية الثلاثة (`verify_projection_orientation`).
  - **`phantom.py`:** فانتوم الصدر ثلاثي الأبعاد غير المتناظر والمسقط الحلزوني التحليلي المغلق وكاتب `DICOM-CT-PD v4`.
  - **تحديث `v0.5.0`:** مطابقة ذكية لأسماء الحالات (`C012` $\leftrightarrow$ `case_C012`) مع إيقاف فوري صريح عند غياب مريض مطلوب (`No silent fallback`) وتوليد تلقائي لصور المقارنة البصرية لـ `Stage 1`.

### ج. المرحلة 2 (`Stage 2 — v0.5.0`): التحويل المتجهي والتدفقي `Curved-to-Flat` + `Helical SSRB` مع `3D FFS` و`measured_mask`
- راجع التوثيق الرياضي والفيزيائي المملّ في [`docs/05_stage2_rebinning_ssrb.md`](./docs/05_stage2_rebinning_ssrb.md):
  1. **`rebin_curved_to_flat` و`CurvedToFlatPlan`:** تقاطع شعاع ثلاثي الأبعاد دقيق من المنبع الفعلي المزاح $\vec{S}(i)$ عبر الكاشف المسطح الممركز إلى الكاشف الفيزيائي (`CYLINDRICAL` / `FLAT` / `SPHERICAL`) مع جداول استيفاء محسوبة مسبقًا لكل حالة `FFS` وقناع صلاحية صارم `valid_flat_mask` (المناطق الخارجة تأخذ `0.0` دون `mode='nearest'`).
  2. **`rebin_helical_ssrb` (Noo et al. 1999 + `3D FFS`):**
     - دعم كلٍّ من `short_scan` ($\pi + 2\gamma_{\max}$ ممركز حول $\phi_{\text{center}}(z_{\text{slice}})$ تمهيدًا لترجيح Parker في `Stage 3`) و`full_scan` ($2\pi$).
     - تعويض دقيق لتغيّر نصف قطر البؤرة $\rho_s^{(g)} = \rho_0 + \Delta\rho^{(g)}$ عبر زاوية الوتر المطابق $\alpha_m^{(g)} = \arcsin\!\bigl(\frac{\rho_0}{\rho_s^{(g)}}\sin\alpha_m\bigr)$ وتقسيم سلس لكل حالة بؤرة $(\Delta\rho, \Delta z)$، مع الترجيح الجيبي التمامي واختيار $\arg\min_{(g,\ell)} |v^{(g,\ell)}|$.
     - دعم `active_row_range` لمحاكاة تقليص صفوف الكاشف (`RDC`) وإظهار أوتاد الفجوات المفقودة في `measured_mask` بصدق فيزيائي كامل.
  3. **`rebin_series_ssrb_streaming`:** نافذة ذاكرة منزلقة محدودة الاستهلاك (`< 350 MiB RAM`) تعيد استخدام المساقط المتداخلة دون إعادة قراءة أي ملف DICOM مرتين وتطابق التشغيل الكامل في الذاكرة بتًا ببت.
  4. **`visual_qa.py`:** توليد تلقائي للوحات المقارنة البصرية متعددة النوافذ بصيغة `PNG` لكلٍّ من بيانات المرضى الحقيقية والفانتوم التحليلي المرجعي (`project_phantom_planar_fanbeam_2d`).

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
│   ├── validation_protocol.md                  # بروتوكول التحقق وبوابة المقارنة البصرية الإلزامية
│   └── decisions/                              # سجلات القرارات المعمارية (ADR-0001 إلى ADR-0004)
├── notebooks/
│   ├── 00_kaggle_stage0.ipynb                  # دفتر Kaggle للمرحلة 0 (التنزيل والفهرسة)
│   ├── 01_kaggle_stage1.ipynb                  # دفتر Kaggle للمرحلة 1 (التدقيق الفيزيائي ومعاينة C012 والفانتوم)
│   └── 02_kaggle_stage2.ipynb                  # دفتر Kaggle للمرحلة 2 (SSRB + عرض لوحات المقارنة البصرية الإلزامية)
├── src/highpitchct/
│   ├── cli.py                                  # واجهة الأوامر الموحدة hpct (شاملة stage1-audit و stage2-rebin)
│   ├── ctpd_reader.py                          # قارئ DICOM-CT-PD الآمن للذاكرة ومحرك براهين التوجيه الثلاثة
│   ├── ctpd_tags.py                            # مفكك الوسوم الثنائية الخاصة وفق DICOM-CT-PD v4 ورابط (0008,9121)
│   ├── geometry.py                             # نموذج الهندسة الحلزونية ثلاثي الأبعاد (HelicalGeometry + 3D FFS)
│   ├── index_dicom.py                          # الفهرسة المتوازية وتدقيق اكتمال السلاسل وتقدير الـ Pitch
│   ├── layout.py                               # مسارات التخزين الحتمية وفصل incoming المؤقت عن raw الدائم
│   ├── manifest.py                             # قراءة وكتابة والتحقق من قوائم CSV للسلاسل
│   ├── phantom.py                              # فانتوم الصدر ثلاثي الأبعاد والإسقاط التحليلي وكاتب DICOM-CT-PD
│   ├── progress.py                             # محرّك أشرطة التقدم الحيّة الآمن للخيوط (tqdm + بديل قياسي)
│   ├── rebinning.py                            # المرحلة 2: Curved-to-Flat و Helical SSRB المتجهي والتدفقي مع 3D FFS
│   ├── runtime.py                              # كاشف البيئة الذكي (kaggle / full) وحارس الميزانية DiskBudgetGuard
│   ├── safe_zip.py                             # استخراج ذري بتمريرة واحدة مع فحص CRC-32 وحماية Zip-Slip
│   ├── tcia_client.py                          # عميل NBIA المتوازي عالي الإنتاجية مع البصمة اللحظية وتقارير السرعة
│   └── visual_qa.py                            # مولّد لوحات المقارنة البصرية الإلزامية بصيغة PNG
└── tests/                                      # حزمة الاختبارات الشاملة (36 اختبارًا تغطي Stage 0 و Stage 1 و Stage 2)
```

وعند التشغيل، تتكون شجرة البيانات والتقارير والأشكال البصرية داخل `<data-root>` كالتالي:

```text
<data-root>/
├── raw/tcia/<case_id>/<role>/<series_uid>/     # ملفات DICOM المستخرجة كما وردت دون أي تعديل
└── manifests/
    ├── figures/
    │   ├── stage1_<case_id>_visual_qa.png      # مقارنة Fortran مقابل C-order وشريحة الصدر المرجعية
    │   ├── stage1_phantom_visual_qa.png        # شرائح ومساقط فانتوم الصدر ثلاثي الأبعاد
    │   ├── stage2_<case_id>_ssrb_visual_qa.png # 6 لوحات: المسقط، الإزاحة، شريحة الصدر، Short/Full SSRB، وفجوات RDC الحمراء
    │   └── stage2_phantom_ssrb_visual_qa.png   # مقارنة SSRB مع السينوغرام المستوي التحليلي الدقيق وخريطة الخطأ وفجوات High-Pitch
    ├── stage1_audit_report.json                # تقرير البراهين الفيزيائية الثلاثة لـ Stage 1 ومسارات الصور البصرية
    └── stage2_rebin_report.json                # تقرير Stage 2 الكمي (نسب التغطية، مطابقة Scalar و2D GT، ومسارات الصور البصرية)
```

---

## 4. أوامر التشغيل والتحديث المباشرة على Kaggle (`Stage 1` المُحدَّث + `Stage 2`)

شغّل الخلايا التالية في دفتر Kaggle (أو افتح الدفتر الجاهز [`notebooks/02_kaggle_stage2.ipynb`](./notebooks/02_kaggle_stage2.ipynb)) لتحديث المستودع إلى `v0.5.0`، تشغيل الـ `36` اختبارًا، تنفيذ التدقيق الكامل على `C012` في المرحلتين الأولى والثانية، وعرض لوحات المقارنة البصرية مباشرةً:

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
!python -m unittest discover -s tests -t . -v
```

```bash
# التأكد من وجود C012 (تخطٍ فوري في 0.0 ثانية إن كانت منزّلة) ثم تشغيل Stage 1 و Stage 2:
!hpct download --series-file configs/pilot_c012.csv --workers 4 --retries 5
!hpct stage1-audit --case-id C012 --run-phantom-check --workers 8
!hpct stage2-rebin --case-id C012 --num-slices 5 --angular-mode short_scan --run-phantom-check --workers 8
```

```python
import json
from IPython.display import Image, Markdown, display

with open("/kaggle/working/highpitchct-data/manifests/stage1_audit_report.json") as f:
    r1 = json.load(f)
with open("/kaggle/working/highpitchct-data/manifests/stage2_rebin_report.json") as f:
    r2 = json.load(f)

for title, path in [
    ("Stage 1 — التحقق البصري للمريض C012", r1["audited_series"][0]["visual_qa_png"]),
    ("Stage 2 — مقارنة SSRB مع الأصل التحليلي ثنائي الأبعاد وفجوات High-Pitch", r2["synthetic_phantom_ssrb_check"]["visual_qa_png"]),
    ("Stage 2 — سينوغرامات C012 الحقيقية وفجوات High-Pitch RDC", r2["rebinned_series"][0]["visual_qa_png"]),
]:
    display(Markdown(f"### {title}"))
    display(Image(filename=path))
```

---

## 5. بوابات القبول المنجزة

### بوابة `Stage 0` (`v0.3.1` — مكتملة ومُثبتة على Kaggle)
- [x] الكشف الذكي عن بيئة `kaggle` (`<= 18.0 GiB` وتخزين `.zip` في `/tmp`) وتنزيل وفهرسة `C012` (`13,446` إسقاطًا + `351` صورة، `VALID`).

### بوابة `Stage 1` (`v0.5.0` — مكتملة ومحققة برمجيًا وبصريًا)
- [x] النموذج الهندسي ثلاثي الأبعاد [`HelicalGeometry`](./src/highpitchct/geometry.py)، القارئ الآمن للذاكرة [`ctpd_reader.py`](./src/highpitchct/ctpd_reader.py)، البراهين الفيزيائية الثلاثة، فانتوم الصدر ثلاثي الأبعاد [`phantom.py`](./src/highpitchct/phantom.py)، مطابقة `C012 <-> case_C012` مع منع التخطي الصامت، وتوليد لوحات المقارنة البصرية.

### بوابة `Stage 2` (`v0.5.0` — مكتملة برمجيًا بـ `36/36` اختبارًا)
- [x] بناء [`rebinning.py`](./src/highpitchct/rebinning.py): تقاطع ثلاثي الأبعاد دقيق `Curved-to-Flat` (`< 1e-13` تطابق شعاعي) ومطابقة تامة مع المرجع القياسي `Scalar` (`< 1e-6`).
- [x] بناء `Helical SSRB` (Noo et al. 1999) مع تعويض كامل للبؤرة المتحركة `3D FFS` ($\Delta\rho, \Delta\phi, \Delta z$) واختيار $\arg\min |v^\star|$ وقناع `measured_mask` الصريح دون أي تمديد زائف `mode='nearest'`.
- [x] إثبات التطابق بتًا ببت بين التدفق محدود الذاكرة `rebin_series_ssrb_streaming` والتشغيل الكامل في الذاكرة، وإثبات مطابقة السينوغرام المستوي التحليلي الدقيق (`MAE < 0.004` على الفانتوم الأسطواني و`< 0.02` على فانتوم الصدر ثلاثي الأبعاد).
- [x] بناء [`visual_qa.py`](./src/highpitchct/visual_qa.py) وتوليد لوحات المقارنة البصرية الإلزامية (`PNG`) لكل مرحلة.

راجع الشرح التفصيلي المملّ لكل خوارزمية ومعادلة وبنية ملفات في مجلد [`docs/`](./docs/).
