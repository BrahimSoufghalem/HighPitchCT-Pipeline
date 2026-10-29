# المرحلة السادسة الموسّعة والمرحلة السابعة (`v1.0.0`): التوسيع الشامل لمحطات العمل (`RTX PRO 6000 96 GB / A100 80 GB`) ودراسة الحذف الكمية السداسية (`Stage 7 Ablation Study`)

## 1. الهدف العلمي والهندسي للإصدار `v1.0.0`

صُمّم الإصدار **`v1.0.0`** للانتقال من تجارب البرهان الأولي محدودة الذاكرة إلى **تدريب بحثي شامل على مستوى محطات العمل الضخمة** (مثل **`NVIDIA RTX PRO 6000 96 GB VRAM`** أو **`A100 80 GB VRAM`** مع **`150 GB CPU RAM`**) دون التفريط بأي قيد فيزيائي من قيود المراحل السابقة:

1. **استخدام كامل الحجم التشريحي لكل مريض (`Full-Volume Patient Slices`):** بدل الاكتفاء بشريحتين لكل مريض، تستخرج الدالة `compute_patient_full_slice_grid` جميع الشرائح الداخلية القابلة لإعادة البناء على امتداد المسح الحلزوني الكامل للمريض ($\sim 235\text{–}245$ شريحة محورية لكل مريض $\times 6$ توليفات من `ITF`/`RDC` و$p \in \{1.5, 2.0, 2.5\}$ $\times 35$ مريضاً $\approx 50,000$ عينة تدريب وتحقق واختبار).
2. **تقسيم ثلاثي صارم خالٍ تماماً من التسرب (`25 Train / 5 Val / 5 Test`):**
   $$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset, \qquad \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \qquad \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset$$
   مع موازنة طبقية (`Stratified Split`) عبر المناطق التشريحية (`Chest` / `Abdomen`).
3. **تسريع شامل لبناء البيانات (`Single-Pass GPU Shard Engine`):**
   - قراءة مساقط `DICOM-CT-PD` الخام مرة واحدة فقط لكل مريض وتحويل `Curved-to-Flat` على البطاقة الرسومية (`torch_cuda`).
   - تنفيذ `SSRB` متعدد الخيوط، ثم `Batched Parker + Ram-Lak FBP` على الـ GPU، ثم الإسقاط الحلزوني ثلاثي الأبعاد الدفعي (`Batched Helical3D`, `view_batch=24` على البطاقات الكبيرة) والإسقاط المروحي ثنائي الأبعاد الدفعي.
   - طباعة حية لنسبة التقدم **`%`** والوقت المتبقي **`ETA`** لكل مريض ولكل خطوة فرعية (`DICOM/C2F`, `SSRB/FBP`, `RDC`, `Helical3D`), مع حفظ ذري (`Atomic _COMPLETE.json`) يسمح باستئناف بناء البيانات فوراً من المريض التالي عند أي انقطاع.
4. **تنشيط الشبكة الثانية `ImageRefineUNet` رباعية القنوات (`4-Ch Activated ImgNet`):**
   - تزويد `ImageRefineUNet` بقناة رابعة صريحة للخطوط التفاضلية:
     $$C_3 = \operatorname{clamp}\!\left(\frac{x_{\text{DC}} - x_{\text{prelim}}}{200\text{ HU}},\, -3,\, +3\right) \odot M_{\text{FOV}}$$
   - تهيئة نشطة لطبقة الإخراج (`0.15` بدل التجميد الصفري شبه التام) ومقياس بواقي `res_scale = 0.50`، مع تخزين مؤقت على القرص لمخرجات `Domain-1` (`x_dc_hu`) خلال `Stage 2` مع تعزيز القلب الإشعاعي ثنائي الأبعاد (`2D Flip Augmentation`).
5. **تدريب `100 Epoch` مسرّع وقابل للاستئناف الذري (`--resume`) مع تقييم `Val` بعد كل Epoch:**
   - دعم الدفعات الحقيقية (`Mini-Batching`, مثل `--batch-size 24`)، وتفعيل `TF32` و`AMP (bfloat16)` داخل التفافات `U-Net` مع إبقاء طبقة `HardDataConsistencyLayer` وجسر `DifferentiableParkerFBPBridge` وحسابات `HU` ودوال الخسارة الفيزيائية بدقة **`float32` صارمة**.
   - تقييم مستقل على مرضى التحقق الخمسة (`Val`) بعد **كل Epoch (`1..100`)** وطباعة: **`RMSE [HU]`, `MAE [HU]`, `PSNR [dB]`, `SSIM`**, ومعدل التعلم `lr`, ونسبة الإنجاز الكلية `%`, والوقت المتبقي المتوقع **`ETA`**.
   - حفظ ذري لملف `latest_training_state.pt` بعد كل Epoch (`model + optimizer + scheduler + scaler + epoch + history + best_val_rmse`) بالإضافة إلى `best_val_dual_domain.pt`.

---

## 2. المرحلة السابعة (`Stage 7`): مصفوفة التقييم الكمي ودراسة الحذف (`M0` $\rightarrow$ `M5`)

تُقارن الوحدة [`ablation.py`](../src/highpitchct/ablation.py) ست طرق متدرجة على شرائح مرضى الاختبار الخمسة غير المرئيين (`Held-Out Unseen Test Patients`) عبر كلا سيناريوهي الفقد (`ITF` و`RDC`) وجميع قيم الـ Pitch السريرية $p \in \{1.5, 2.0, 2.5\}$:

| الرمز | الطريقة | المجال | الوصف العلمي والغرض في مذكرة الماستر |
| :---: | :--- | :---: | :--- |
| **`M0`** | `Raw No-Fill FBP` | تحليلي | إعادة بناء `Parker + Ram-Lak FBP` مباشرة مع ترك الفجوة الحلزونية صفراً (خط الأساس الأسوأ لتبيان شدة التشوه) |
| **`M1`** | `Angular Feather FBP` | تحليلي | استكمال خطي زاوي بسيط للفجوة المفقودة قبل `FBP` |
| **`M2`** | `Conjugate-Ray Prelim FBP` | فيزيائي تحليلي | استكمال الأشعة المرافقة الفيزيائي (`x_prelim_hu`) — خط الأساس الفيزيائي القوي |
| **`M3`** | `Image-Only U-Net` | تعلم عميق (صورة فقط) | شبكة `ImageRefineUNet` مستقلة مدربة مباشرة لتحويل $x_{\text{prelim}} \to x_{\text{target}}$ دون أي شبكة سينوغرام |
| **`M4`** | `Sino-Only DL (SinoNet + HardDC + FBP)` | تعلم عميق (سينوغرام + فيزياء) | المجال الأول فقط مع فرض الاتساق الصارم على الأشعة المقاسة ($\text{Max Dev} = 0.0$) وجسر `Parker Ram-Lak FBP` |
| **`M5`** | `Full Dual-Domain DL` | تعلم عميق ثنائي المجال | النموذج الكامل المقترح (`SinoNet + HardDC + Differentiable FBP Bridge + 4-Ch Activated ImgNet`) |

---

## 3. التشغيل القياسي الشامل على محطة العمل (`CLI`)

```bash
# 1) اكتشاف 35 مريضاً (صدر + بطن) من مجموعة TCIA LDCT-and-Projection-data
hpct discover-tcia \
  --body-part CHEST,ABDOMEN \
  --roles projection_full,reconstruction_published \
  --max-cases 35 \
  --output /workspace/highpitchct-data/manifests/cohort_35_patients.csv

# 2) تنزيل السلاسل بملف تعريف محطة العمل الكامل (بدون سقف 18 GiB الخاص بـ Kaggle)
hpct download \
  --series-file /workspace/highpitchct-data/manifests/cohort_35_patients.csv \
  --data-root /workspace/highpitchct-data \
  --profile full \
  --cleanup-zip \
  --workers 8

# 3) بناء الشظايا السريع لكل شرائح الـ 35 مريضاً (~50,000 عينة) + تدريب 100 Epoch مع Val + تشغيل Stage 7 Ablation
hpct stage6-scale \
  --data-root /workspace/highpitchct-data \
  --n-train 25 --n-val 5 --n-test 5 \
  --target-total-samples 50000 \
  --pitches 1.5,2.0,2.5 \
  --scenario both \
  --image-size 256 \
  --epochs 100 \
  --batch-size 24 \
  --lr 1.5e-3 \
  --backend torch_cuda \
  --amp bf16 \
  --workers 12 \
  --num-loader-workers 6 \
  --run-stage7-ablation \
  --m3-epochs 15 \
  --resume
```
