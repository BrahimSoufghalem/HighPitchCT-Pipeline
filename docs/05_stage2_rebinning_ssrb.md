# التوثيق الرياضي والفيزيائي والبرمجي التفصيلي للمرحلة الثانية (`Stage 2 — v0.5.0`): التحويل من كاشف منحنٍ إلى مسطح (`Curved-to-Flat`) وإعادة الترتيب الحلزوني أحادي الشريحة (`SSRB`) مع معالجة البؤرة المتحركة ثلاثية الأبعاد (`3D FFS`) وقناع القياس الصريح (`measured_mask`)

تُمثّل **المرحلة الثانية (`Stage 2`)** الجسر الهندسي والفيزيائي الحاسم الذي ينقل مساقط الأشعة المخروطية الحلزونية الخام (`3D Helical Cone-Beam Projections`) المقروءة في `Stage 1` إلى **سينوغرامات مراوح مستوية ثنائية الأبعاد (`2D Planar Fan-Beam Sinograms`)** عند مواضع الشرائح المحورية المطلوبة $z_{\text{slice}}$، مع مرافقة كل سينوغرام بقناع منطقي صريح `measured_mask` يحدّد بدقة تامة كل بكسل مقاس فيزيائيًا مقابل كل بكسل مفقود بسبب الـ `High-Pitch`.

---

## 1. الأخطاء الفيزيائية والبرمجية الأربع في الأدوات المنشورة سابقًا (`Helix2Fan`) وكيف صحّحناها جذريًا

أثناء الفحص الدقيق لملف `rebinning_functions.py` في المستودع المرجعي السابق (`Helix2Fan`)، تبيّن وجود أربع مغالطات هندسية خطيرة كانت ستُفسد تدريب الشبكة العصبية لو نُقلت كما هي:

| رقم الخلل | ما كان يفعله الكود المرجعي السابق (`Helix2Fan`) | الأثر الفيزيائي الفعلي على بيانات `TCIA / C012` | الحل الجذري المنفَّذ في [`rebinning.py`](../src/highpitchct/rebinning.py) |
| :---: | :--- | :--- | :--- |
| **1** | استخدام معيار كروي $\sqrt{u^2 + d_0^2 + v^2}$ بدل المعيار الأسطواني الأفقي $\sqrt{u^2 + d_0^2}$ عند إسقاط اللوح المسطح على كاشف **أسطواني (`CYLINDRICAL`)** | تشويه الإحداثي المحوري $v$ عند أطراف المروحة بمقدار زائف لا ينتمي لهندسة أسطوانة جهاز `Siemens SOMATOM Definition AS+` | حلّ معادلة تقاطع الشعاع ثلاثي الأبعاد مع السطح الفيزيائي الحقيقي (`CYLINDRICAL` أو `FLAT` أو `SPHERICAL`) صراحةً من المبادئ الأولى في [`_solve_ray_detector_hit_local`](../src/highpitchct/rebinning.py) |
| **2** | حساب العمود المركزي بالصيغة الخاطئة `nu - det_central_element[0]` (`736 - 369.625 = 366.375` بدل `c0 = 368.625`، أي خطأ `2.25` عمود $\approx 2.89\text{ mm}$) مع **إهمال تام للبؤرة المتحركة `3D FFS`** | قفز متناوب في مسار الأوتار الشعاعية بين المساقط الزوجية والفردية بمقدار يصل إلى **$1.64\text{ mm}$** عند أضلاع الصدر بسبب $\Delta\rho = +5.45\text{ mm}$ في وضع `FFSZ` | تعويض هندسي ثلاثي الأبعاد كامل لـ $(\Delta\rho, \Delta\phi, \Delta z)$ في كلٍّ من `Curved-to-Flat` و`SSRB` مع مطابقة وتر الشعاع الأفقي بدقة عبر $\alpha_m^{(g)} = \arcsin\!\bigl(\frac{\rho_0}{\rho_s^{(g)}}\sin\alpha_m\bigr)$ |
| **3** | تمرير `mode='nearest'` داخل `scipy.ndimage.map_coordinates` دون أي قناع بيانات مفقودة | تكرار صفوف حافة الكاشف (`Row 0` و`Row 63`) عبر الفجوات المفقودة في الـ `High-Pitch`، مما يخلق خطوطًا زائفة ويخفي موضع النقص الحقيقي عن مرحلة `Extrapolation` وعن الشبكة العصبية | حظر `mode='nearest'` نهائيًا؛ رفض كل شعاع يقع خارج `[0, N_rows - 1]` أو خارج النطاق الفعّال `active_row_range`، وتعيين `measured_mask = False` وقيمة `0.0` صراحةً |
| **4** | الكتابة الفوقية العشوائية (`overwrite`) عند تداخل دورتين حلزونيتين بحسب ترتيب حلقة `for` | استبدال شعاع قريب من المستوى المركزي للكاشف (زاوية مخروطية صغيرة) بشعاع بعيد من دورة مجاورة ذي تشوه مخروطي أكبر | مفاضلة صريحة عبر جميع حالات البؤرة `FFS` وجميع الدورات $\ell \in \mathbb{Z}$ باختيار القياس ذي المسافة المحورية الصغرى **$\arg\min_{(g,\ell)} \|v^{(g,\ell)}\|$** |

---

## 2. الاشتقاق الرياضي الكامل للخطوة الأولى: `rebin_curved_to_flat`

### 2.1 تعريف الكاشف المسطح الافتراضي الممركز (`Virtual Centered Flat Detector`)
عند المسقط $i$، يقع مركز بؤرة الكاشف الفيزيائي عند $\vec{F}_0(i) = (-\rho_0\sin\phi_{0,i},\; +\rho_0\cos\phi_{0,i},\; z_{0,i})$ بأساس متعامد نظامي $(\hat{e}_{\text{ray}}(\phi_{0,i}),\; \hat{e}_u(\phi_{0,i}),\; \hat{e}_z)$، بينما تقع البؤرة الفعلية للأشعة السينية بعد إزاحات `3D FFS` عند:

$$\rho_{s,i} = \rho_0 + \Delta\rho_i, \qquad \phi_{s,i} = \phi_{0,i} + \Delta\phi_i, \qquad z_{s,i} = z_{0,i} + \Delta z_i$$

$$\vec{S}(i) = \bigl(-\rho_{s,i}\sin\phi_{s,i},\; +\rho_{s,i}\cos\phi_{s,i},\; z_{s,i}\bigr)$$

نُعرّف لكل مسقط $i$ **كاشفًا مسطحًا افتراضيًا ممركزًا** متعامدًا مع الشعاع المركزي الفعلي $\hat{e}_{\text{ray}}(\phi_{s,i})$ على مسافة $d_0$ من المنبع الفعلي $\vec{S}(i)$، أبعاده $(N_v, N_u)$ وخطوتاه $(d_v, d_u)$، وإحداثياته الممركزة (مع الحفاظ على اتفاقية `Row 0 = +z`):

$$u_m = \left(m - \frac{N_u - 1}{2}\right) d_u \quad (m = 0,\dots,N_u-1), \qquad v_n = \left(\frac{N_v - 1}{2} - n\right) d_v \quad (n = 0,\dots,N_v-1)$$

بما أن جسم المريض يقع بالكامل حول محور الدوران بين المنبع $\vec{S}(i)$ وكلا الكاشفين (الافتراضي والفيزيائي) وسط الهواء ($\mu = 0$)، فإن التكامل الخطي $\int \mu(\vec{x})\,\mathrm{d}\ell$ على طول نصف المستقيم المنطلق من $\vec{S}(i)$ عبر البكسل الافتراضي $(n, m)$ **يساوي تمامًا** قيمة التكامل الخطي عند نقطة تقاطع نصف المستقيم نفسه مع سطح الكاشف الفيزيائي $(r_{\text{hit}}, c_{\text{hit}})$.

### 2.2 معادلات التقاطع الدقيق في جملة الكاشف الفيزيائي المحلية
بكتابة متجه إزاحة المنبع $\vec{q} = \vec{S}(i) - \vec{F}_0(i)$ ومتجه اتجاه الشعاع $\vec{w}(n, m) = d_0\hat{e}_{\text{ray}}(\phi_{s,i}) + u_m\hat{e}_u(\phi_{s,i}) + v_n\hat{e}_z$ في جملة الكاشف الفيزيائي $(\hat{e}_{\text{ray}}(\phi_{0,i}),\; \hat{e}_u(\phi_{0,i}),\; \hat{e}_z)$ حيث $\Delta\phi_i = \phi_{s,i} - \phi_{0,i}$:

$$q_{\text{ray}} = \rho_0 - \rho_{s,i}\cos\Delta\phi_i, \qquad q_u = -\rho_{s,i}\sin\Delta\phi_i, \qquad q_z = \Delta z_i$$

$$w_{\text{ray}} = d_0\cos\Delta\phi_i - u_m\sin\Delta\phi_i, \qquad w_u = d_0\sin\Delta\phi_i + u_m\cos\Delta\phi_i, \qquad w_z = v_n$$

نجد وسيط التقاطع الموجب $t > 0$ للشعاع $\vec{x}(t) = \vec{q} + t\vec{w}$ مع سطح الكاشف الفيزيائي كما يلي:

1. **للكاشف الأسطواني (`CYLINDRICAL` — كما في `TCIA / Siemens Definition AS+`):**
   معادلة الأسطوانة $x_{\text{ray}}^2 + x_u^2 = d_0^2$ تعطي:
   $$A = w_{\text{ray}}^2 + w_u^2, \qquad B = q_{\text{ray}}w_{\text{ray}} + q_u w_u, \qquad C = q_{\text{ray}}^2 + q_u^2 - d_0^2, \qquad t = \frac{-B + \sqrt{B^2 - AC}}{A}$$
   $$\gamma_{\text{hit}} = \operatorname{atan2}(q_u + t w_u,\; q_{\text{ray}} + t w_{\text{ray}}), \qquad c_{\text{hit}} = c_0 + \frac{d_0\,\gamma_{\text{hit}}}{d_u}, \qquad r_{\text{hit}} = r_0 - \frac{q_z + t w_z}{d_v}$$
2. **للكاشف المسطح (`FLAT`):**
   $$t = \frac{d_0 - q_{\text{ray}}}{w_{\text{ray}}}, \qquad c_{\text{hit}} = c_0 + \frac{q_u + t w_u}{d_u}, \qquad r_{\text{hit}} = r_0 - \frac{q_z + t w_z}{d_v}$$
3. **للكاشف الكروي (`SPHERICAL`):**
   $$A = \|\vec{w}\|^2, \qquad B = \vec{q}\cdot\vec{w}, \qquad C = \|\vec{q}\|^2 - d_0^2, \qquad t = \frac{-B + \sqrt{B^2 - AC}}{A}$$
   $$\gamma_{\text{hit}} = \operatorname{atan2}(x_u^{\text{hit}}, x_{\text{ray}}^{\text{hit}}), \quad \alpha_{\text{hit}} = \arcsin\!\left(\frac{x_z^{\text{hit}}}{d_0}\right), \quad c_{\text{hit}} = c_0 + \frac{d_0\gamma_{\text{hit}}}{d_u}, \quad r_{\text{hit}} = r_0 - \frac{d_0\alpha_{\text{hit}}}{d_v}$$

يُحسب جدول الاستيفاء ثنائي الخطية `(r0, r1, c0, c1, w00, w01, w10, w11)` **مرةً واحدة فقط لكل حالة فريدة من حالات `FFS`** في [`CurvedToFlatPlan`](../src/highpitchct/rebinning.py) (حالتان فقط في `FFSZ`!)، ويُرفق بقناع الصلاحية الصارم `valid_flat_mask` بحيث يأخذ كل بكسل خارج حدود الكاشف القيمة `0.0` دون أي تمديد حواف.

---

## 3. الاشتقاق الرياضي الكامل للخطوة الثانية: `rebin_helical_ssrb` (طريقة Noo et al. 1999 مع تعويض `3D FFS`)

### 3.1 تعريف شبكة السينوغرام المستوي المستهدف لكل شريحة $z_{\text{slice}}$
لإعادة بناء شريحة مقطعية عند $z = z_{\text{slice}}$ على دائرة منبع اسمية نصف قطرها $\rho_0$ وكاشف مسطح ممركز على بُعد $d_0$:
- زاوية نصف المروحة القصوى: $\gamma_{\max} = \arctan\!\left(\frac{(N_u - 1)d_u}{2 d_0}\right)$ (في `C012`: $\gamma_{\max} = 23.52^\circ$).
- زاوية مرور المنبع بمركز الشريحة $\phi_{\text{center}}(z_{\text{slice}})$ تُستخرج بالانحدار الخطي الدقيق لـ $z_s(\phi_s)$.
- شبكة الزوايا $\beta_k$ تُبنى **متزايدة رتيبًا دائمًا** ($\Delta\beta > 0$) بغض النظر عن اتجاه دوران الجهاز:
  - في وضع **`short_scan`** (الوضع القياسي لـ `High-Pitch CT` متوافقًا مع ترجيح Parker في `Stage 3`): المدى الزاوي هو $\Theta_{\text{short}} = \pi + 2\gamma_{\max}$ (`227.05°` في `C012`) ممركز حول $\phi_{\text{center}}(z_{\text{slice}})$:
    $$\beta_k \in \left[\phi_{\text{center}}(z_{\text{slice}}) - \frac{\pi + 2\gamma_{\max}}{2},\; \phi_{\text{center}}(z_{\text{slice}}) + \frac{\pi + 2\gamma_{\max}}{2}\right]$$
  - في وضع **`full_scan`**: المدى الزاوي هو $2\pi$ (`360°`):
    $$\beta_k \in \bigl[\phi_{\text{center}}(z_{\text{slice}}) - \pi,\; \phi_{\text{center}}(z_{\text{slice}}) + \pi\bigr)$$

### 3.2 مطابقة وتر الشعاع الأفقي تحت تغيّر نصف قطر البؤرة $\rho_s^{(g)} = \rho_0 + \Delta\rho^{(g)}$
في السينوغرام المستهدف، يُحدّد الزوج $(\beta_k, u_m)$ حيث $\alpha_m = \arctan(u_m / d_0)$ خطًا أفقيًا في المستوي $(x, y)$ زاوية انتشاره $\theta = \beta_k + \alpha_m$ وبعده العمودي الموقّع عن مبدأ الإحداثيات $t_{\text{radon}} = \rho_0 \sin\alpha_m$.

عندما تكون البؤرة الفعلية في الحالة $g$ على نصف قطر $\rho_s^{(g)} = \rho_0 + \Delta\rho^{(g)}$ (مثل $\rho_s^{(1)} = 600.45\text{ mm}$ مقابل $\rho_0 = 595.0\text{ mm}$ في `C012`)، فإن تطابق المستقيم الأفقي ذاته في $(x, y)$ يفرض شرطين هندسيين صارمين:

$$\rho_s^{(g)} \sin\alpha_m^{(g)} = \rho_0 \sin\alpha_m \implies \alpha_m^{(g)} = \arcsin\!\left(\frac{\rho_0}{\rho_s^{(g)}} \sin\alpha_m\right)$$

$$u_m^{(g)} = d_0 \tan\alpha_m^{(g)}, \qquad c_m^{(g)} = \frac{u_m^{(g)}}{d_u} + \frac{N_u - 1}{2}$$

$$\phi_s^\star(\beta_k, m; \ell) = \beta_k + \bigl(\alpha_m - \alpha_m^{(g)}\bigr) + 2\pi\ell, \qquad \ell \in \mathbb{Z}$$

### 3.3 تقسيم حالات البؤرة `FFS` وحساب إحداثي Noo المحوري والترجيح الجيبي التمامي
تُقسَّم المساقط بواسطة [`_partition_ffs_view_groups`](../src/highpitchct/rebinning.py) تبعًا للزوج الفيزيائي $(\Delta\rho, \Delta z)$ بحيث يمتلك كل قسم $g$ نصف قطر ثابتًا $\rho_s^{(g)}$ ومسارًا حلزونيًا أملس $z_s^{(g)}(\phi_s)$ خاليًا من أي تذبذب منشاري، مع الاستفادة الكاملة من مضاعفة المعاينة المحورية لبؤرة `z-FFS`.

عند استيفاء $z_s^\star$ خطيًا بين المسقطين المحيطين $(j_0, j_1)$ ضمن الحالة $g$ عند الزاوية $\phi_s^\star$، يُفرض تقاطع الشعاع المائل مع مستوي الشريحة $z = z_{\text{slice}}$ عند منتصف وتر الشعاع داخل ساحة الرؤية ($s_{xy,\text{mid}} = \rho_s^{(g)}\cos\alpha_m^{(g)}$) وفق معادلة Noo et al. (1999):

$$v^{(g,\ell)} = \bigl(z_{\text{slice}} - z_s^\star\bigr)\,\frac{d_0^2 + \bigl(u_m^{(g)}\bigr)^2}{\rho_s^{(g)}\,d_0}, \qquad r^{(g,\ell)} = \frac{N_v - 1}{2} - \frac{v^{(g,\ell)}}{d_v}$$

ويُضرب التكامل الخطي المستوفى ثلاثيًا في معامل تصحيح ميل الشعاع المخروطي (`Cosine Foreshortening Weight`):

$$w_{\cos}^{(g,\ell)} = \frac{\sqrt{d_0^2 + \bigl(u_m^{(g)}\bigr)^2}}{\sqrt{d_0^2 + \bigl(u_m^{(g)}\bigr)^2 + \bigl(v^{(g,\ell)}\bigr)^2}}$$

ومن بين جميع المرشحين الصالحين $(g, \ell)$ الذين يقعون داخل مدى الزوايا المقاسة وداخل صفوف الكاشف الفعّالة $[r_{\min}, r_{\max}]$ وتكون جميع أركان الاستيفاء الثمانية صالحة في `valid_flat_mask`، نختار القياس الأقرب للمستوى المركزي للكاشف:

$$(g^\star, \ell^\star) = \underset{(g,\ell)\text{ صالح}}{\arg\min}\;\bigl|v^{(g,\ell)}\bigr|$$

أما إذا لم يوجد أي مرشح صالح (كما يحدث في فجوات الـ `High-Pitch`)، فتُسجَّل النقطة فورًا `measured_mask = False` و`sinogram = 0.0`.

---

## 4. التدفق بالنافذة المنزلقة محدود الذاكرة (`rebin_series_ssrb_streaming`) لبيئة Kaggle

تحتوي سلسلة المريض `C012` على `13,446` مسقطًا بحجم خام يقارب `2.53 GiB` في صيغة `float32`. ولتشغيل `Stage 2` بسرعة فائقة وذاكرة عشوائية منخفضة جدًا (`< 350 MiB RAM`) على بيئة `Kaggle` (`30 GB RAM` وسقف قرص `<= 18 GiB`):

1. تحسب الدالة [`rebin_series_ssrb_streaming`](../src/highpitchct/rebinning.py) زوايا المراكز $\phi_{\text{center}}(z_{\text{slice}})$ من الهندسة الكاملة للسلسلة لضمان تطابق الشبكة الزاوية `beta_rad` تطابقًا تامًا (بتًا ببت) مع التشغيل الكامل في الذاكرة.
2. تُرتّب الشرائح المطلوبة تصاعديًا في $z$ وتُقسّمها إلى دفعات مكانية متقاربة (`max_batch_z_span_mm = 20 mm`)، وتحدد لكل دفعة مدى المساقط اللازم فيزيائيًا فقط استنادًا إلى المتراجحة الصارمة:
   $$|z_s(i) - z_{\text{slice}}| \le v_{\max}\frac{\rho_{s,\max}}{d_0}$$
3. تحتفظ بنافذة ذاكرة منزلقة (`Sliding Window Cache`) تُعيد استخدام المساقط المشتركة بين الدفعات المتتالية بحيث **لا يُقرأ أي ملف DICOM ولا يُحوّل من منحنٍ إلى مسطح أكثر من مرة واحدة إطلاقًا**، مع تحرير المساقط الخلفية فور تجاوزها.

---

## 5. بوابة المقارنة البصرية الإلزامية (`visual_qa.py`)

تطبيقًا للقاعدة الدستورية في [`docs/validation_protocol.md`](./validation_protocol.md)، تُولّد وحدة [`visual_qa.py`](../src/highpitchct/visual_qa.py) تلقائيًا لوحات مقارنة بصرية بصيغة `PNG` داخل `<data-root>/manifests/figures/`:

1. **`stage1_phantom_visual_qa.png` و`stage1_<case_id>_visual_qa.png`:**
   مقارنة بصرية مباشرة بين شرائح الفانتوم الأصلية ومساقطه، وبين تخطيط `Fortran (pixel_array.T)` الفيزيائي السليم وتخطيط `C-order` المشوَّه والشريحة المرجعية المنشورة للمريض.
2. **`stage2_phantom_ssrb_visual_qa.png` (أربع لوحات مقارنة مرجعية):**
   - **اللوحة A:** السينوغرام المستوي ثنائي الأبعاد التحليلي الدقيق (`Exact 2D Planar Fan-Beam Ground Truth`) عند $z = 0\text{ mm}$.
   - **اللوحة B:** سينوغرام `Curved-to-Flat + SSRB` الناتج من مسار حلزوني ثلاثي الأبعاد مع `FFSZ` عند `pitch = 0.85`.
   - **اللوحة C:** خريطة الخطأ المطلق $|g_{\text{SSRB}} - g_{\text{exact}}|$ موثّقةً بقيمتَي `MAE` و`RMSE`.
   - **اللوحة D:** سينوغرام `High-Pitch` (`pitch = 1.85`) مع تلوين الفجوات غير المقاسة (`~measured_mask`) باللون الأحمر الصريح لإثبات غياب أي تمديد زائف `mode='nearest'`.
3. **`stage2_<case_id>_ssrb_visual_qa.png` (ست لوحات على بيانات المريض الحقيقية مثل `C012`):**
   تعرض المسقط الأسطواني الخام، خريطة الإزاحة الهندسية `|Flat - Curved|`، شريحة الصدر المرجعية عند نفس الموضع $z_{\text{mid}}$، سينوغرام `Short-Scan` عند الـ Pitch الأصلي (`0.90`، تغطية `~100%`)، سينوغرام `Full-Scan` ($360^\circ$)، وسينوغرام محاكاة الـ `High-Pitch RDC` بنصف صفوف الكاشف (`32` صفًا، `effective pitch = 1.79`) مع ظهور أوتاد البيانات المفقودة باللون الأحمر بوضوح تام.

---

## 6. أوامر التشغيل وبوابة قبول المرحلة الثانية (`hpct stage2-rebin`)

```bash
# 1. تشغيل المرحلة الأولى المُحدَّثة (تتضمن إصلاح مطابقة C012 <-> case_C012 وتوليد صور المقارنة البصرية):
hpct stage1-audit --case-id C012 --run-phantom-check --workers 8

# 2. تشغيل المرحلة الثانية الكاملة على C012 + الفانتوم المرجعي وحفظ السينوغرامات وصور المقارنة البصرية:
hpct stage2-rebin --case-id C012 --num-slices 5 --angular-mode short_scan --run-phantom-check --workers 8
```
