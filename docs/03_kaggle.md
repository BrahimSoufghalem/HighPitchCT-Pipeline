# التشغيل على Kaggle

## الخلاصة

بيئة **T4 x2** مناسبة لبناء نسخة الماستر واختبارها وتدريب النموذج على patches، بشرط ألا نحمل كل الإسقاطات أو الحجم ثلاثي الأبعاد في RAM/VRAM مرة واحدة.

## ما تعنيه الموارد فعليًا

- البطاقتان منفصلتان؛ ذاكرة GPU الثانية لا تُضاف تلقائيًا إلى الأولى.
- يجب افتراض قرابة 15 GB قابلة للاستخدام لكل T4، لا كتلة واحدة بحجم 30 GB.
- وثائق Kaggle تعرض قرابة 29 GB من RAM لنمط T4 x2.
- `/kaggle/input` للقراءة فقط.
- `/kaggle/working` هو مكان الكتابة، ومساحته المحفوظة محدودة؛ لذلك القرص قد يصبح القيد الأول قبل GPU.
- تنزيل TCIA ونسخ GitHub يحتاجان تفعيل Internet في إعدادات Notebook.

## الإعدادات المقترحة

### Stage 0: التنزيل والفهرسة

لا تحتاج GPU. يفضّل Accelerator=None إذا كنت لا تريد استهلاك حصة GPU. فعّل Internet، ونزّل حالة Pilot واحدة فقط.

### الهندسة وSSRB وFBP

استخدم T4 واحدة أولًا للتحقق. نقسم الحساب حسب slices أو مجموعات views، ونكتب النتيجة على القرص تدريجيًا. لا نبقي كل الإسقاطات والنتائج والـintermediate tensors معًا.

### تدريب النموذج

- mixed precision بواسطة AMP.
- patch training بدل volumes كاملة.
- batch صغير مع gradient accumulation.
- `DistributedDataParallel` عندما نقرر استخدام البطاقتين؛ وجود T4 x2 لا يجعل PyTorch يستخدمهما تلقائيًا.
- كل worker يقرأ جزءه فقط من البيانات.

## تشغيل Stage 0

في أول خلية:

```bash
!git clone https://github.com/BrahimSoufghalem/HighPitchCT-Pipeline.git
%cd HighPitchCT-Pipeline
!python -m pip install -q -e .
!hpct env
```

المسار الافتراضي يصبح تلقائيًا:

```text
/kaggle/working/highpitchct-data
```

يمكن تغييره:

```python
import os
os.environ["HPCT_DATA_ROOT"] = "/kaggle/working/my-hpct-data"
```

لا تضبطه إلى `/kaggle/input` لأنه read-only.

## إدارة المساحة

أثناء التنزيل يوجد ZIP والنسخة المستخرجة معًا، أي أن الاستهلاك المؤقت قد يقترب من ضعفي حجم السلسلة. بعد التحقق وحفظ hash يمكن حذف ZIP يدويًا لتوفير المساحة، لكن لا نحذف تلقائيًا في Stage 0 حتى لا نخفي مشكلة أو نفقد provenance.

للتجارب الكبيرة:

1. أنشئ Kaggle Dataset خاصًا/عامًا للبيانات المفهرسة المسموح بإعادة توزيعها.
2. أرفقه بالـNotebook ليظهر تحت `/kaggle/input`.
3. اجعل المخرجات فقط في `/kaggle/working`.
4. احفظ checkpoints وتقارير القبول بانتظام؛ لا تعتمد على بقاء الجلسة.

## اختبار سريع للبيئة

```bash
!hpct env
!python -m unittest discover -s tests -v
```

راجع `disk_free_bytes` قبل تنزيل أي Series، وراجع `gpus` قبل التدريب.
