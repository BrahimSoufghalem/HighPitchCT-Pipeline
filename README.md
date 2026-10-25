# HighPitchCT-Pipeline

إطار بحثي وهندسي محكم وقابل للتدقيق لإعادة بناء وتقييم نموذج **ثنائي المجال (Dual-Domain Deep Learning)** لإعادة بناء التصوير المقطعي الحلزوني مرتفع الـ Pitch (**High-Pitch Helical CT**) انطلاقًا من بيانات الإسقاط الخام في مجموعة **TCIA (`LDCT-and-Projection-data`)**، ومُصمَّم خصيصًا لمذكرة ماستر في **الفيزياء الطبية** وفق نظام بوابات المراحل الصارم (**Stage-Gate**).

> **الإصدار الحالي:** `v0.4.0` — **المرحلة 0 (`Stage 0`) مكتملة ومُثبتة على Kaggle** + **المرحلة 1 (`Stage 1`) مكتملة ومحققة بـ `30/30` اختبارًا**: النموذج الهندسي الحلزوني ثلاثي الأبعاد الصريح (`HelicalGeometry`) مع البؤرة المتحركة ثلاثية الأبعاد (`3D FFS`)، قارئ `DICOM-CT-PD v4` الآمن للذاكرة (`order='F'` $\equiv$ `pixel_array.T` دون تكرار اللوغاريتم مع تدفق `chunks` و`np.memmap`)، البراهين الفيزيائية الكمية الثلاثة للتوجيه (`verify_projection_orientation`)، والفانتوم التحليلي ثلاثي الأبعاد للصدر (`3D Thorax Phantom`) مع الإسقاط الأمامي التحليلي المغلق.

---

## 0. خارطة المراحل الكاملة للمشروع (`Stage 0` $\rightarrow$ `Stage 9`)

| المرحلة | الاسم والمحتوى العلمي والهندسي | الحالة | المخرجات وبوابة القبول |
| :--- | :--- | :---: | :--- |
| **`Stage 0`** | اكتشاف وتنزيل `TCIA` فائق السرعة الواعي ببيئة `Kaggle <= 18 GiB` + فهرسة رؤوس `DICOM-CT-PD v4` | ✅ **مكتملة ومُثبتة على Kaggle (`v0.3.1`)** | `13,446` إسقاطًا + `351` صورة للمريض `C012` بحالة `VALID` (`pitch = 0.8966`) |
| **`Stage 1`** | الهندسة الحلزونية ثلاثية الأبعاد (`3D FFS`) + قارئ `DICOM-CT-PD` الآمن للذاكرة + 3 براهين توجيه فيزيائية + فانتوم الصدر ثلاثي الأبعاد | ✅ **مكتملة برمجيًا (`v0.4.0`) — بانتظار تأكيد تشغيل `stage1-audit` على `C012` في Kaggle** | `geometry.py`, `ctpd_reader.py`, `phantom.py`, `hpct stage1-audit`, `30/30` اختبارًا ناجحًا |
| **`Stage 2`** | إعادة الترتيب الهندسي المتجهي السريع: `Curved-to-Flat` + `SSRB` مع معالجة `3D FFS` وقناع القياس الصريح `measured_mask` | ⏭ **التالية مباشرةً** | إنتاج سينوغرامات مراوح مستوية لكل شريحة $z$ مع خريطة الفجوات الحقيقية دون `nearest` زائف |
| **`Stage 3`** | استكمال الجوانب (`Extrapolation`) + ترجيح `Short-scan / Parker` + `FBP` القياسي وإنتاج المرجع $x_{\text{reference}}$ | ⏳ قادمة | إعادة بناء مرجعية نظيفة في جملة إحداثياتنا الموحدة وقياس `PSNR / SSIM / RMSE` على الفانتوم و`C012` |
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
| **منهجية العمل** | نظام **Stage-Gate** صارم: كود + توثيق تفصيلي مملّ + 4 مستويات اختبار (Unit / Synthetic / Integration / Real-data) لكل مرحلة | منع تراكم الأخطاء الهندسية وعدم الانتقال لأي مرحلة تالية قبل اجتياز بوابة القبول |
| **إدارة الـ Backends** | تحديد صريح للـ Backend في الإعدادات **ومنْع أي انتقال صامت (`No silent fallback`)** | ضمان قابلية تكرار التجارب وعدم خلط خوارزميات إعادة بناء مختلفة بين المرضى |
| **الوعي الذكي بالبيئة** | اكتشاف تلقائي للبيئة (`kaggle` مقابل `full`): على Kaggle يلتزم بسقف **$\le 18.0\text{ GiB}$** في `/kaggle/working` ويضع الـ `.zip` المؤقت في `/tmp` ويحذفه بعد الاستخراج؛ وخارج Kaggle ينزّل كامل القائمة | تمكين التجريب الآمن والسريع لـ $10\text{–}12$ مريضًا على Kaggle ($2\times\text{T4}$)، ثم تشغيل الكود نفسه دون تعديل على خادم أكبر لكامل البيانات |
| **قراءة DICOM-CT-PD** | فكّ ثنائي صريح للوسوم الخاصة (`VR=UN`) وفق دليل `DICOM-CT-PD v4`، ترتيب وفحص استمرارية الزاوية و$z_0$، وتحويل المصفوفة بـ `pixel_array.T` (`order='F'`) إلى `(64, 736)` دون تكرار اللوغاريتم (`LogFlag=YES`) | ملفات TCIA مكتوبة بـ `Implicit VR Little Endian` وبترتيب أعمدة MATLAB (`Rows=736, Columns=64`)؛ أي قراءة بـ `order='C'` تخلط الصفوف بالأعمدة |
| **تقدير الـ Pitch والهندسة** | حساب ديناميكي لكل سلسلة عبر انحدار خطي لـ $z_0(\phi_{\text{unwrap}})$ لاستخراج `table_feed_mm_per_rot` و`normalized_pitch` واتجاهي الدوران والطاولة | عدم افتراض ثبات الـ pitch عند `0.9` أو `0.6` أو افتراض إشارة دوران موجبة دائمًا (في `C012` تتناقص $\phi_0$ بمقدار $-\frac{2\pi}{1152}$ لكل view) |
| **البؤرة المتحركة (`FFS`)** | قراءة وتطبيق المركبات الثلاث $(\rho_0+\Delta\rho,\; \phi_0+\Delta\phi,\; z_0+\Delta z)$ لكل view | في جهاز `SOMATOM Definition AS+` حتى عند `FFSZ` تنزاح البؤرة شعاعيًا بـ $\Delta\rho = +5.45\text{ mm}$ بسبب ميل المصعد ($\approx 6.9^\circ$) |
| **الربط بين الصور والإسقاطات** | استخدام تسلسل `ReferenceRawDataSequence (0008,9121)` $\rightarrow$ `(0008,1115)` $\rightarrow$ `(0020,000E)` | في بيانات TCIA يختلف كلٌّ من `StudyInstanceUID` و`FrameOfReferenceUID` بين الصور المنشورة والإسقاطات الخام |
| **Curved-to-Flat & SSRB** | تحويل متجهي سريع مع `measured_mask` صريح وإلغاء `mode='nearest'` خارج الكاشف، واختيار القياس الأقرب لمركز الكاشف ($\arg\min_k \|v_k^\star\|$)، وفصل `Extrapolation` كمرحلة مستقلة | تمثيل نقص البيانات الفعلي الناتج عن الـ high-pitch بأمانة بدل إخفائه بتكرار أطراف الكاشف |
| **المرجع (`Reference Reconstruction`)** | إعادة بناء $x_{\text{reference}}$ من الإسقاطات الخام كاملة الجرعة عند الـ pitch الأصلي لكل فحص عبر خط `SSRB + Short-scan + Parker + FBP` الخاص بنا | منع الشبكة من تعلم تقليد مرشح إعادة البناء التجاري (`B50f` مثلًا) وإزاحات إحداثياته، وحصر التعلم في إزالة تشوهات الـ high-pitch |
| **محاكاة ITF وRDC والتعلم** | تحويل $x_{\text{reference}}$ إلى $\mu(x,y,z)$ وتوليد إسقاطات أمامية حلزونية (`LEAP`) لحالات ITF/RDC، وتدريب شبكتي Residual U-Net مرتبطتين عبر `Differentiable FBP` بخسارة $\mathcal{L}_1 + \mathcal{L}_\nabla$ | مطابقة الفيزياء الفعلية لمسار الطاولة وتطبيق البنية الرياضية الموصوفة في ملصق AAPM 2026 |

---

## 2. تفاصيل المراحل المنفذة (`Stage 0` + `Stage 1`)

### أ. المرحلة 0 (`Stage 0 — v0.3.1`): التنزيل المتوازي الذكي والفهرسة السريعة
- **`runtime.py`:** الكشف التلقائي عن بيئة `kaggle` (سقف صارم `<= 18.0 GiB` في `/kaggle/working/highpitchct-data` وتخزين مؤقت للـ `.zip` في `/tmp/highpitchct-incoming`) مقابل وضع `full` غير المقيد على الحواسيب والخوادم.
- **`tcia_client.py` و`safe_zip.py` و`progress.py`:** جلسات `requests.Session` مجمّعة لكل خيط مع كتل تدفق `4 MiB`، بصمة `SHA-256` أثناء التدفق، استخراج ذري بتمريرة واحدة مع فحص `CRC-32` وحماية `Zip-Slip`، أشرطة تقدم `tqdm` حيّة، وتقرير سرعة `download_report.csv`.
- **`ctpd_tags.py` و`index_dicom.py`:** فكّ ثنائي دقيق للوسوم الخاصة `VR=UN` وفهرسة متوازية (`--workers 8`) مع فحص اكتمال الزوايا وكشف البتر المحوري.

### ب. المرحلة 1 (`Stage 1 — v0.4.0`): الهندسة الحلزونية ثلاثية الأبعاد، القارئ الآمن للذاكرة، البراهين الفيزيائية للتوجيه، والفانتوم التحليلي
راجع التوثيق الرياضي والبرمجي المملّ في [`docs/04_stage1_reader_and_phantom.md`](./docs/04_stage1_reader_and_phantom.md):
1. **`geometry.py` (`HelicalGeometry`):**
   - نموذج ثلاثي الأبعاد صريح للنظام الأيسر $(x = -\rho\sin\phi,\; y = +\rho\cos\phi,\; z)$ وفق دليل `DICOM-CT-PD v4` (الصفحات 3–5).
   - فصل تام بين مركز بؤرة الكاشف $\vec{F}_0(i)$ وموضع المنبع الفعلي المزاح بالبؤرة المتحركة ثلاثية الأبعاد $\vec{S}(i) = (\rho_0+\Delta\rho_i,\; \phi_{0,i}+\Delta\phi_i,\; z_{0,i}+\Delta z_i)$.
   - تحويل صريح لمركز الكاشف `(7031,1033)` من ترقيم 1 (`[369.625, 32.5]`) إلى ترقيم 0 (`(368.625, 31.5)`).
   - حساب الإحداثيات الديكارتية الدقيقة لكل عنصر كاشف $\vec{D}(r, c; i)$ للأشكال `CYLINDRICAL` و`FLAT` و`SPHERICAL` مع اتجاه الصفوف `Row 0 = +z` ($v(r) = (r_0 - r)d_v$).
2. **`ctpd_reader.py`:**
   - قراءة صفرية النسخ عبر `np.frombuffer(..., dtype="<u2").reshape((n_rows, n_cols), order="F")` المطابقة لـ `ds.pixel_array.T`، مع التحجيم الخطي لـ `float32` واحترام `LogFlag=YES` دون تكرار اللوغاريتم.
   - تدفق متوازٍ على دفعات خفيفة (`iter_projection_chunks` بذاكرة `< 50 MiB`) وتفريغ مباشر إلى `np.memmap` على القرص (`load_projections_memmap`).
   - تحويل ثنائي الاتجاه محكم بين `HU` و$\mu\;[\text{mm}^{-1}]$ (`hu_to_mu`, `mu_to_hu`) وقارئ حجوم الصور المنشورة `read_ct_image_volume` للتحقق البصري فقط.
   - **محرّك البراهين الفيزيائية الثلاثة (`verify_projection_orientation`):**
     1. نعومة التدرج المحوري `order='F'` مقابل `order='C'`،
     2. **مبرهنة تطابق الأشعة المترافقة الحلزونية ثلاثية الأبعاد (`Exact 3D Helical Conjugate-Ray Identity`)** التي تختبر في آنٍ واحد إشارة الصفوف (`row 0 = +z` مقابل `-z`)، إشارة الأعمدة (`+\hat{e}_u` مقابل `-\hat{e}_u`)، وطرح `1.0` من مركز الكاشف وإزاحة ربع العنصر،
     3. تداخل الدورة الكاملة $2\pi$ عند مركز الدوران عندما $p < 0.96$.
3. **`phantom.py`:**
   - فانتوم صدر إهليلجي ثلاثي الأبعاد كاسر للتناظر على المحاور الثلاثة $(x, y, z)$ (`build_thorax_phantom` و`voxelize_phantom`).
   - مسقط أمامي مخروطي حلزوني تحليلي مغلق (`project_phantom_helical`) يدعم `FFSZ` بالكامل ومُطابق ضد التكامل العددي الدقيق بـ `40,000` خطوة على طول الشعاع.
   - كاتب ملفات `DICOM-CT-PD v4` اصطناعية كاملة (`write_synthetic_ctpd_series`) لاختبارات الذهاب والإياب الحتمية (`RMSE = 2.55e-5`).

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
│   ├── validation_protocol.md                  # بروتوكول التحقق وبوابات القبول الصارمة
│   └── decisions/                              # سجلات القرارات المعمارية (ADR-0001 إلى ADR-0004)
├── notebooks/
│   ├── 00_kaggle_stage0.ipynb                  # دفتر Kaggle للمرحلة 0 (التنزيل والفهرسة)
│   └── 01_kaggle_stage1.ipynb                  # دفتر Kaggle للمرحلة 1 (التدقيق الفيزيائي ومعاينة C012 والفانتوم)
├── src/highpitchct/
│   ├── cli.py                                  # واجهة الأوامر الموحدة hpct (شاملة stage1-audit)
│   ├── ctpd_reader.py                          # قارئ DICOM-CT-PD الآمن للذاكرة ومحرك براهين التوجيه الثلاثة
│   ├── ctpd_tags.py                            # مفكك الوسوم الثنائية الخاصة وفق DICOM-CT-PD v4 ورابط (0008,9121)
│   ├── geometry.py                             # نموذج الهندسة الحلزونية ثلاثي الأبعاد (HelicalGeometry + 3D FFS)
│   ├── index_dicom.py                          # الفهرسة المتوازية وتدقيق اكتمال السلاسل وتقدير الـ Pitch
│   ├── layout.py                               # مسارات التخزين الحتمية وفصل incoming المؤقت عن raw الدائم
│   ├── manifest.py                             # قراءة وكتابة والتحقق من قوائم CSV للسلاسل
│   ├── phantom.py                              # فانتوم الصدر ثلاثي الأبعاد والإسقاط التحليلي وكاتب DICOM-CT-PD
│   ├── progress.py                             # محرّك أشرطة التقدم الحيّة الآمن للخيوط (tqdm + بديل قياسي)
│   ├── runtime.py                              # كاشف البيئة الذكي (kaggle / full) وحارس الميزانية DiskBudgetGuard
│   ├── safe_zip.py                             # استخراج ذري بتمريرة واحدة مع فحص CRC-32 وحماية Zip-Slip
│   └── tcia_client.py                          # عميل NBIA المتوازي عالي الإنتاجية مع البصمة اللحظية وتقارير السرعة
└── tests/                                      # حزمة الاختبارات الشاملة (30 اختبارًا تغطي Stage 0 و Stage 1)
```

وعند التشغيل، تتكون شجرة البيانات والتقارير داخل `<data-root>` كالتالي:

```text
<data-root>/
├── raw/tcia/<case_id>/<role>/<series_uid>/     # ملفات DICOM المستخرجة كما وردت دون أي تعديل
└── manifests/
    ├── downloads/<series_uid>.json             # سجل كل سلسلة (الحالة، الحجم، السرعة، SHA-256، الاستخراج)
    ├── download_report.csv                     # جدول سرعات التنزيل والأحجام لكل سلسلة (MiB/s)
    ├── download_run_summary.json               # ملخص تشغيل التنزيل والسرعة المتوسطة واستهلاك حصة القرص
    ├── dicom_index.csv                         # فهرس تفصيلي لكل ملف DICOM على حدة
    ├── series_inventory.csv                    # جدول الفيزياء والتدقيق لكل سلسلة (Pitch، تغطية، حالة القبول)
    ├── dicom_index_summary.json                # التقرير الشامل للفهرسة وزمن التنفيذ والربط بين السلاسل
    └── stage1_audit_report.json                # تقرير البراهين الفيزيائية الثلاثة لـ Stage 1 وفحص الفانتوم ثلاثي الأبعاد
```

---

## 4. دليل الأوامر الكامل (`hpct`)

### أ. فحص البيئة وحصة القرص والبطاقات الرسومية
```bash
hpct env
# أو لاختبار محاكاة وضع Kaggle محليًا:
hpct env --profile kaggle --data-root ./test_kaggle_data
```

### ب. اكتشاف وتنزيل وفهرسة سلاسل TCIA (`Stage 0`)
```bash
hpct validate-manifest --series-file configs/pilot_c012.csv
hpct download --series-file configs/pilot_c012.csv --workers 4 --retries 5
hpct index --workers 8
```

### ج. التدقيق الفيزيائي والهندسي الكامل لـ `Stage 1` (`hpct stage1-audit`)
```bash
# تدقيق المريض الحقيقي C012 + فحص الذهاب والإياب على الفانتوم ثلاثي الأبعاد:
hpct stage1-audit --case-id C012 --run-phantom-check --workers 8

# تشغيل فحص الفانتوم التحليلي ثلاثي الأبعاد مستقلًا دون الحاجة لأي بيانات خارجية:
hpct stage1-audit --run-phantom-check --data-root ./tmp_stage1_check
```

---

## 5. أوامر التشغيل والتحديث المباشرة على Kaggle (`Stage 1`)

شغّل الخلايا التالية في دفتر Kaggle لتحديث المستودع إلى `v0.4.0`، تشغيل الـ `30` اختبارًا، وتنفيذ التدقيق الفيزيائي الكامل على المريض `C012` (يمكنك أيضًا فتح الدفتر الجاهز [`notebooks/01_kaggle_stage1.ipynb`](./notebooks/01_kaggle_stage1.ipynb)):

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
# إذا لم تكن بيانات C012 منزّلة في الجلسة الحالية سينزّلها بسرعة، وإن كانت موجودة سيتخطاها فورًا:
!hpct download --series-file configs/pilot_c012.csv --workers 4 --retries 5

# تشغيل بوابة التدقيق الفيزيائي للمرحلة 1 على C012 + الفانتوم ثلاثي الأبعاد:
!hpct stage1-audit --case-id C012 --run-phantom-check --workers 8
```

---

## 6. بوابات القبول المنجزة

### بوابة `Stage 0` (`v0.3.1` — مكتملة ومُثبتة على Kaggle)
- [x] الكشف الذكي عن بيئة `kaggle` مقابل `full`، وفرض سقف `18.0 GiB` وتفريغ `.zip` المؤقت في `/tmp` على Kaggle.
- [x] فكّ جميع الوسوم الخاصة `VR=UN` وفق `DICOM-CT-PD v4` وحساب الـ Pitch بالانحدار الخطي دون تحميل `PixelData`.
- [x] التحقق الفعلي على Kaggle للمريض الكامل `C012` (`13,446` ملف إسقاط + `351` صورة، `pitch = 0.8966`، الحالة `VALID`) مع تسريع الشبكة وأشرطة تقدم `tqdm`.

### بوابة `Stage 1` (`v0.4.0` — مكتملة برمجيًا بـ `30/30` اختبارًا)
- [x] بناء [`HelicalGeometry`](./src/highpitchct/geometry.py) بنظام الإحداثيات الأيسر الصريح، دعم كامل للبؤرة المتحركة ثلاثية الأبعاد $(\Delta\rho, \Delta\phi, \Delta z)$، وتحويل `0-based` لمركز الكاشف.
- [x] بناء [`ctpd_reader.py`](./src/highpitchct/ctpd_reader.py) بالقراءة الصفرية النسخ `order='F'` المطابقة لـ `pixel_array.T`، احترام `LogFlag=YES`، والتدفق الخفيف للذاكرة (`chunks` و`memmap`).
- [x] اجتياز البراهين الفيزيائية الكمية الثلاثة (`Fortran vs C`، مبرهنة تطابق الأشعة المترافقة الحلزونية ثلاثية الأبعاد، وتداخل الدورة الكاملة $2\pi$) ورفض السلاسل التالفة صراحةً (`No silent fallback`).
- [x] بناء الفانتوم التحليلي ثلاثي الأبعاد غير المتناظر للصدر [`phantom.py`](./src/highpitchct/phantom.py) ومطابقة مسقطه التحليلي المغلق ضد التكامل العددي بـ `40,000` خطوة واجتياز اختبار الذهاب والإياب لملفات `.dcm` (`RMSE = 2.55e-5`).

راجع الشرح التفصيلي المملّ لكل خوارزمية ومعادلة وبنية ملفات في مجلد [`docs/`](./docs/).
