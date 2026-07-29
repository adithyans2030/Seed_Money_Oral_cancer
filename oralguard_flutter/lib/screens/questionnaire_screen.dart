import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/nav_bar.dart';
import '../widgets/clinician_bottom_sheet.dart';
import '../models/questionnaire_models.dart';
import '../models/questionnaire_data.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class QuestionnaireScreen extends StatefulWidget {
  const QuestionnaireScreen({super.key});

  @override
  State<QuestionnaireScreen> createState() => _QuestionnaireScreenState();
}

class _QuestionnaireScreenState extends State<QuestionnaireScreen> {
  late List<QuestionSection> _sections;
  RiskResult? _result;
  final _scrollController = ScrollController();
  final _resultKey = GlobalKey();

  @override
  void initState() {
    super.initState();
    _sections = buildSections();
  }

  double get _progress {
    int total = 0, answered = 0;
    for (final s in _sections) {
      for (final q in s.questions) {
        total++;
        if (q.answer != AnswerValue.unanswered) answered++;
        if (q.showDiff) {
          for (final d in q.diffQuestions) {
            total++;
            if (d.answer != AnswerValue.unanswered) answered++;
          }
        }
      }
    }
    return total == 0 ? 0 : answered / total;
  }

  void _setAnswer(QuestionItem q, AnswerValue val) {
    setState(() {
      q.answer = val;
      if (val == AnswerValue.no) {
        for (final d in q.diffQuestions) {
          d.answer = AnswerValue.unanswered;
        }
      }
      _result = null;
    });
  }

  void _setDiffAnswer(DiffQuestion d, AnswerValue val) =>
      setState(() => d.answer = val);

  Future<void> _evaluate() async {
    final prefs = await SharedPreferences.getInstance();
    final cbirSessionId = prefs.getString('cbir_session_id');

    final allQ = _sections.expand((s) => s.questions).toList();
    final allD = allQ.expand((q) => q.diffQuestions).toList();
    
    bool isYes(String id) => 
      allQ.any((q) => q.id == id && q.answer == AnswerValue.yes) ||
      allD.any((d) => d.id == id && d.answer == AnswerValue.yes);

    final body = jsonEncode({
      "answers": {
        "age": 35, // default
        "gender": 1, // default male
        "region": "Unknown",
        "q_tobacco": isYes('c1') ? 1 : 0,
        "q_alcohol": isYes('c4') ? 1 : 0,
        "q_hpv": isYes('c5') ? 1 : 0,
        "q_betel": isYes('c3') ? 1 : 0,
        "q_sun": isYes('c6') ? 1 : 0,
        "q_hygiene": 0,
        "q_diet": 0,
        "q_family": isYes('c7') ? 1 : 0,
        "q_immune": isYes('c9') ? 1 : 0,
        "q_lesions": (isYes('s1') || isYes('c8')) ? 1 : 0,
        "q_bleeding": 0,
        "q_swallowing": isYes('s5') ? 1 : 0,
        "q_patches": isYes('s2') ? 1 : 0,
        "questionnaire_raw": {}
      },
      "session_id": cbirSessionId
    });

    try {
      final res = await http.post(
        Uri.parse('https://gprabhanjana-oral-cancer-cbir-api.hf.space/predict-risk'),
        headers: {'Content-Type': 'application/json'},
        body: body,
      );
      
      if (res.statusCode == 200) {
         final data = jsonDecode(res.body);
         final riskLabel = data['risk_label'] as String;
         final riskScore = data['risk_score'] as double;
         
         RiskResult result;
         if (riskLabel == 'High') {
            result = RiskResult(
              level: RiskLevel.high,
              title: 'High Risk — Seek Immediate Consultation',
              body: 'Your AI assessed risk score is ${(riskScore*100).toStringAsFixed(1)}%.\n\nPlease consult a dentist, oral surgeon, or oncologist as soon as possible — ideally within the next 1–2 weeks. Do not wait for symptoms to worsen.',
              bullets: [
                'Book an urgent appointment with an oral health specialist',
                'Bring a list of all medications, habits, and symptom duration',
              ],
            );
         } else if (riskLabel == 'Moderate') {
            result = RiskResult(
              level: RiskLevel.caution,
              title: 'Elevated Risk — Caution Advised',
              body: 'Your AI assessed risk score is ${(riskScore*100).toStringAsFixed(1)}%.\n\nWhile the absence of symptoms is reassuring, your risk profile warrants attention. Schedule a routine oral cancer screening with your dentist if you have not had one in the past year.',
              bullets: [
                'Schedule an oral cancer screening examination',
                'Discuss your risk factors openly with your dentist or GP',
              ],
            );
         } else {
            result = RiskResult(
              level: RiskLevel.none,
              title: 'No Immediate Concerns Identified',
              body: 'Your AI assessed risk score is ${(riskScore*100).toStringAsFixed(1)}%.\n\nBased on your responses, you have not reported significant risk factors or active symptoms associated with oral cancer at this time.',
              bullets: [
                'Attend regular dental check-ups at least once a year',
                'Maintain a healthy diet and limit alcohol consumption',
              ],
            );
         }
         
         setState(() => _result = result);
         
         if (cbirSessionId != null) {
           _checkCombinedRisk(cbirSessionId);
         }
         
         WidgetsBinding.instance.addPostFrameCallback((_) {
            if (_resultKey.currentContext != null) {
              Scrollable.ensureVisible(_resultKey.currentContext!,
                  duration: const Duration(milliseconds: 500),
                  curve: Curves.easeInOut);
            }
         });
         return;
      }
    } catch (e) {
      debugPrint("API failed: $e");
    }

    _fallbackEvaluate();
  }

  Future<void> _checkCombinedRisk(String sessionId) async {
    try {
      final res = await http.post(
        Uri.parse('https://gprabhanjana-oral-cancer-cbir-api.hf.space/combined-risk'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({"session_id": sessionId}),
      );
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        if (data['combined_risk_label'] != null) {
          // Just appending it to the existing body for simplicity
          setState(() {
            if (_result != null) {
              _result = RiskResult(
                level: _result!.level,
                title: _result!.title,
                body: '${_result!.body}\n\nCOMBINED TRIAGE ASSESSMENT:\n${data['combined_risk_label']} - ${data['recommendation']}',
                bullets: _result!.bullets,
                diffNote: _result!.diffNote,
              );
            }
          });
        }
      }
    } catch (e) {
      debugPrint("Combined risk failed: $e");
    }
  }

  void _fallbackEvaluate() {
    const causeIds   = ['c1','c2','c3','c4','c5','c6','c7','c8','c9'];
    const symptomIds = ['s1','s2','s3','s4','s5','s6','s7','s8','s9'];
    const diffIds    = ['d1','d2','d3','d4','d5','d6','d7','d8'];

    final allQ = _sections.expand((s) => s.questions).toList();
    final allD = allQ.expand((q) => q.diffQuestions).toList();

    bool hasCause   = allQ.where((q) => causeIds.contains(q.id)).any((q) => q.answer == AnswerValue.yes);
    bool hasSymptom = allQ.where((q) => symptomIds.contains(q.id)).any((q) => q.answer == AnswerValue.yes);
    bool diffPresent = allD.where((d) => diffIds.contains(d.id)).any((d) => d.answer == AnswerValue.yes);

    RiskResult result;

    if (hasCause && hasSymptom) {
      result = RiskResult(
        level: RiskLevel.high,
        title: 'High Risk — Seek Immediate Consultation',
        body: 'You have reported both significant risk factors and active symptoms associated with oral cancer. This combination warrants urgent clinical evaluation.\n\nPlease consult a dentist, oral surgeon, or oncologist as soon as possible — ideally within the next 1–2 weeks. Do not wait for symptoms to worsen.\n\nEarly-stage oral cancer is highly treatable. Prompt action is the most important thing you can do right now.',
        diffNote: diffPresent
            ? 'Some of your symptoms may overlap with other conditions based on your follow-up answers. A clinician will rule out alternatives through examination and biopsy if required — this does not reduce the urgency of being seen.'
            : null,
        bullets: [
          'Book an urgent appointment with an oral health specialist',
          'Bring a list of all medications, habits, and symptom duration',
          'Do not attempt to self-treat or ignore lesions',
        ],
      );
    } else if (hasCause && !hasSymptom) {
      result = RiskResult(
        level: RiskLevel.caution,
        title: 'Elevated Risk — Caution Advised',
        body: 'You have identified one or more significant risk factors for oral cancer but have not reported active symptoms at this time.\n\nWhile the absence of symptoms is reassuring, your risk profile warrants attention. Schedule a routine oral cancer screening with your dentist if you have not had one in the past year.\n\nReducing or eliminating modifiable risks — especially tobacco, areca nut, and alcohol — can significantly lower your lifetime risk.',
        bullets: [
          'Schedule an oral cancer screening examination',
          'Discuss your risk factors openly with your dentist or GP',
          'Consider cessation support if substance use is a factor',
          'Perform monthly self-examinations and report any new changes lasting over 3 weeks',
        ],
      );
    } else if (!hasCause && hasSymptom && !diffPresent) {
      result = RiskResult(
        level: RiskLevel.consult,
        title: 'Symptoms Present — Consultation Recommended',
        body: 'You have reported symptoms consistent with oral cancer, even without identified lifestyle risk factors. Oral cancer can occur in individuals with no obvious risk history.\n\nWe recommend consulting a dentist or oral health specialist to have your symptoms properly evaluated. Most oral lesions are benign, but persistent or unexplained changes always warrant professional assessment.',
        bullets: [
          'Book a dental or oral medicine consultation',
          'Note the duration, location, and character of each symptom to share with your clinician',
          'If any symptom has persisted beyond 3 weeks, prioritise this appointment',
        ],
      );
    } else if (!hasCause && hasSymptom && diffPresent) {
      result = RiskResult(
        level: RiskLevel.diffDiagnosis,
        title: 'Possible Alternate Diagnosis — Consultation Still Advised',
        body: 'Your symptom responses are consistent with oral cancer presentations, but your follow-up answers suggest these may also be explained by other, non-cancerous conditions — such as aphthous ulcers, oral candidiasis, lichen planus, traumatic lesions, or reactive lymphadenopathy.\n\nThis is important context, but it does not eliminate the need for professional evaluation. Differential diagnoses can only be confirmed — and malignancy ruled out — through direct clinical examination and biopsy where indicated.',
        bullets: [
          'Schedule a dental or oral medicine consultation',
          'Share the specific locations, duration, and any associated triggers with your clinician',
          'If a lesion has not resolved within 3 weeks of removing an apparent cause, escalate urgency',
        ],
      );
    } else {
      result = RiskResult(
        level: RiskLevel.none,
        title: 'No Immediate Concerns Identified',
        body: 'Based on your responses, you have not reported significant risk factors or active symptoms associated with oral cancer at this time.\n\nThis is encouraging — however, good oral health habits remain important for everyone.',
        bullets: [
          'Attend regular dental check-ups at least once a year',
          'Maintain a healthy diet and limit alcohol consumption',
          'Avoid tobacco and areca nut products',
          'Perform periodic self-examinations of your mouth',
          'Report any changes that persist beyond 3 weeks to your dentist',
        ],
      );
    }

    setState(() => _result = result);

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_resultKey.currentContext != null) {
        Scrollable.ensureVisible(_resultKey.currentContext!,
            duration: const Duration(milliseconds: 500),
            curve: Curves.easeInOut);
      }
    });
  }

  void _reset() {
    setState(() {
      _sections = buildSections();
      _result = null;
    });
    _scrollController.animateTo(0,
        duration: const Duration(milliseconds: 500), curve: Curves.easeInOut);
  }

  @override
  Widget build(BuildContext context) {
    final isMobile = Bp.isMobile(context);
    final fs = fontScale(context);
    final hPad = isMobile ? 16.0 : 20.0;

    return Scaffold(
      appBar: const OralGuardNavBar(),
      bottomNavigationBar: isMobile ? const OralGuardBottomNav() : null,
      body: SingleChildScrollView(
        controller: _scrollController,
        padding: EdgeInsets.symmetric(horizontal: hPad, vertical: 24),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 720),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Masthead
                Text('CLINICAL SCREENING TOOL',
                    style: GoogleFonts.sourceSans3(
                        fontSize: 9 * fs,
                        letterSpacing: 3.5,
                        color: AppColors.rust,
                        fontWeight: FontWeight.w600)),
                const SizedBox(height: 8),
                Text('Oral Cancer\nRisk Questionnaire',
                    style: GoogleFonts.playfairDisplay(
                        fontSize: 30 * fs, height: 1.15)),
                const SizedBox(height: 12),
                Text(
                  'This screener evaluates risk factors and symptoms associated with oral cancer. Answer each question honestly. Results are indicative only.',
                  style: GoogleFonts.sourceSans3(
                      fontSize: 13 * fs, color: AppColors.muted, height: 1.7),
                ),
                const SizedBox(height: 14),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: const BoxDecoration(
                    color: AppColors.rustLight,
                    borderRadius: BorderRadius.only(
                      topRight: Radius.circular(4),
                      bottomRight: Radius.circular(4),
                    ),
                    border: Border(
                        left: BorderSide(color: AppColors.rust, width: 3)),
                  ),
                  child: Text(
                    '⚠ This tool is not a substitute for professional clinical evaluation. Always consult a qualified dentist or oncologist.',
                    style: GoogleFonts.sourceSans3(
                        fontSize: 12 * fs, color: AppColors.rust),
                  ),
                ),
                const SizedBox(height: 20),

                // Self-exam trigger
                _SelfExamTrigger(),
                const SizedBox(height: 6),

                // Progress bar
                LinearProgressIndicator(
                  value: _progress,
                  backgroundColor: AppColors.border,
                  valueColor: const AlwaysStoppedAnimation(AppColors.rust),
                  minHeight: 3,
                  borderRadius: BorderRadius.circular(2),
                ),
                const SizedBox(height: 20),

                // Sections
                ..._sections.map((s) => _SectionWidget(
                      section: s,
                      onAnswer: _setAnswer,
                      onDiffAnswer: _setDiffAnswer,
                    )),
                const SizedBox(height: 20),

                // Submit — full-width on mobile
                SizedBox(
                  width: isMobile ? double.infinity : null,
                  child: Center(
                    child: ElevatedButton(
                      onPressed: _evaluate,
                      style: ElevatedButton.styleFrom(
                        padding: EdgeInsets.symmetric(
                          horizontal: isMobile ? 24 : 48,
                          vertical: 16,
                        ),
                      ),
                      child: Text('EVALUATE MY RISK',
                          style: GoogleFonts.sourceSans3(
                              fontSize: 14 * fs,
                              fontWeight: FontWeight.w600,
                              letterSpacing: 1)),
                    ),
                  ),
                ),
                const SizedBox(height: 28),

                // Result
                if (_result != null) ...[
                  Container(
                      key: _resultKey,
                      child: _ResultCard(result: _result!, onReset: _reset)),
                ],
                const SizedBox(height: 40),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ─── SELF-EXAM TRIGGER ────────────────────────────────────────────────────────

class _SelfExamTrigger extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final fs = fontScale(context);
    return GestureDetector(
      onTap: () => context.push('/self-exam'),
      child: Container(
        padding: const EdgeInsets.all(14),
        margin: const EdgeInsets.only(bottom: 14),
        decoration: BoxDecoration(
          color: AppColors.white,
          border: Border.all(color: AppColors.border),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Row(
          children: [
            const Text('🔍', style: TextStyle(fontSize: 22)),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Before you begin — do a quick self-exam',
                      style: GoogleFonts.sourceSans3(
                          fontSize: 13 * fs, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 2),
                  Text(
                      'Open the visual guide to check your mouth first, then answer the questions below.',
                      style: GoogleFonts.sourceSans3(
                          fontSize: 11 * fs, color: AppColors.muted)),
                ],
              ),
            ),
            const Icon(Icons.arrow_outward, color: AppColors.border),
          ],
        ),
      ),
    );
  }
}

// ─── SECTION ──────────────────────────────────────────────────────────────────

class _SectionWidget extends StatelessWidget {
  final QuestionSection section;
  final Function(QuestionItem, AnswerValue) onAnswer;
  final Function(DiffQuestion, AnswerValue) onDiffAnswer;

  const _SectionWidget({
    required this.section,
    required this.onAnswer,
    required this.onDiffAnswer,
  });

  @override
  Widget build(BuildContext context) {
    final fs = fontScale(context);
    return Container(
      margin: const EdgeInsets.only(bottom: 18),
      decoration: BoxDecoration(
        color: AppColors.white,
        border: Border.all(color: AppColors.border),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Container(
                  width: 34,
                  height: 34,
                  decoration: const BoxDecoration(
                    color: Color(0xFFFEF3E2),
                    shape: BoxShape.circle,
                  ),
                  child: Center(
                      child:
                          Text(section.icon, style: const TextStyle(fontSize: 15))),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(section.title,
                          style:
                              GoogleFonts.playfairDisplay(fontSize: 16 * fs)),
                      Text(section.subtitle,
                          style: GoogleFonts.sourceSans3(
                              fontSize: 11 * fs, color: AppColors.muted)),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          ...section.questions.expand((q) => [
                _QuestionRow(question: q, onAnswer: (val) => onAnswer(q, val)),
                if (q.showDiff)
                  _DiffBlock(question: q, onDiffAnswer: onDiffAnswer),
              ]),
        ],
      ),
    );
  }
}

// ─── QUESTION ROW ─────────────────────────────────────────────────────────────

class _QuestionRow extends StatelessWidget {
  final QuestionItem question;
  final Function(AnswerValue) onAnswer;

  const _QuestionRow({required this.question, required this.onAnswer});

  @override
  Widget build(BuildContext context) {
    final isMobile = Bp.isMobile(context);
    final fs = fontScale(context);

    // On very narrow screens, stack question text above toggle
    if (isMobile) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: const BoxDecoration(
          border: Border(bottom: BorderSide(color: Color(0xFFF0ECE6))),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(question.text,
                style:
                    GoogleFonts.sourceSans3(fontSize: 13 * fs, height: 1.5)),
            if (question.hint != null) ...[
              const SizedBox(height: 3),
              Text(question.hint!,
                  style: GoogleFonts.sourceSans3(
                      fontSize: 11 * fs, color: AppColors.muted)),
            ],
            const SizedBox(height: 10),
            _ToggleGroup(value: question.answer, onChanged: onAnswer),
          ],
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: Color(0xFFF0ECE6))),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(question.text,
                    style: GoogleFonts.sourceSans3(
                        fontSize: 14 * fs, height: 1.5)),
                if (question.hint != null) ...[
                  const SizedBox(height: 4),
                  Text(question.hint!,
                      style: GoogleFonts.sourceSans3(
                          fontSize: 12 * fs, color: AppColors.muted)),
                ],
              ],
            ),
          ),
          const SizedBox(width: 12),
          _ToggleGroup(value: question.answer, onChanged: onAnswer),
        ],
      ),
    );
  }
}

// ─── DIFF BLOCK ───────────────────────────────────────────────────────────────

class _DiffBlock extends StatelessWidget {
  final QuestionItem question;
  final Function(DiffQuestion, AnswerValue) onDiffAnswer;

  const _DiffBlock({required this.question, required this.onDiffAnswer});

  @override
  Widget build(BuildContext context) {
    final isMobile = Bp.isMobile(context);
    final fs = fontScale(context);

    return Container(
      decoration: const BoxDecoration(
        color: Color(0xFFF5FAF8),
        border: Border(
          left: BorderSide(color: AppColors.sage, width: 3),
          bottom: BorderSide(color: Color(0xFFF0ECE6)),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            color: AppColors.sageLight,
            child: Text(
              '🔬  Follow-up — Differentiating the symptom',
              style: GoogleFonts.sourceSans3(
                fontSize: 11 * fs,
                fontWeight: FontWeight.w600,
                color: AppColors.sage,
                letterSpacing: 0.5,
              ),
            ),
          ),
          ...question.diffQuestions.map((d) => Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 16, vertical: 12),
                decoration: const BoxDecoration(
                  border:
                      Border(bottom: BorderSide(color: Color(0xFFE0EDE9))),
                ),
                child: isMobile
                    ? Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(d.text,
                              style: GoogleFonts.sourceSans3(
                                  fontSize: 12 * fs, height: 1.5)),
                          if (d.hint != null) ...[
                            const SizedBox(height: 3),
                            Text(d.hint!,
                                style: GoogleFonts.sourceSans3(
                                    fontSize: 10 * fs,
                                    color: AppColors.muted)),
                          ],
                          const SizedBox(height: 8),
                          _ToggleGroup(
                            value: d.answer,
                            onChanged: (val) => onDiffAnswer(d, val),
                            small: true,
                          ),
                        ],
                      )
                    : Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(d.text,
                                    style: GoogleFonts.sourceSans3(
                                        fontSize: 13 * fs, height: 1.5)),
                                if (d.hint != null) ...[
                                  const SizedBox(height: 3),
                                  Text(d.hint!,
                                      style: GoogleFonts.sourceSans3(
                                          fontSize: 11 * fs,
                                          color: AppColors.muted)),
                                ],
                              ],
                            ),
                          ),
                          const SizedBox(width: 12),
                          _ToggleGroup(
                            value: d.answer,
                            onChanged: (val) => onDiffAnswer(d, val),
                            small: true,
                          ),
                        ],
                      ),
              )),
        ],
      ),
    );
  }
}

// ─── TOGGLE GROUP ─────────────────────────────────────────────────────────────

class _ToggleGroup extends StatelessWidget {
  final AnswerValue value;
  final Function(AnswerValue) onChanged;
  final bool small;

  const _ToggleGroup(
      {required this.value, required this.onChanged, this.small = false});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _ToggleBtn(
          label: 'Yes',
          isActive: value == AnswerValue.yes,
          activeColor: AppColors.rust,
          onTap: () => onChanged(AnswerValue.yes),
          small: small,
        ),
        const SizedBox(width: 6),
        _ToggleBtn(
          label: 'No',
          isActive: value == AnswerValue.no,
          activeColor: AppColors.ink,
          onTap: () => onChanged(AnswerValue.no),
          small: small,
        ),
      ],
    );
  }
}

class _ToggleBtn extends StatelessWidget {
  final String label;
  final bool isActive;
  final Color activeColor;
  final VoidCallback onTap;
  final bool small;

  const _ToggleBtn({
    required this.label,
    required this.isActive,
    required this.activeColor,
    required this.onTap,
    this.small = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        // Larger touch target on mobile
        padding: EdgeInsets.symmetric(
          horizontal: small ? 12 : 16,
          vertical: 9,
        ),
        decoration: BoxDecoration(
          color: isActive ? activeColor : AppColors.white,
          border: Border.all(color: isActive ? activeColor : AppColors.border),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Text(
          label,
          style: GoogleFonts.sourceSans3(
            fontSize: small ? 12 : 13,
            fontWeight: isActive ? FontWeight.w600 : FontWeight.w300,
            color: isActive ? Colors.white : AppColors.muted,
          ),
        ),
      ),
    );
  }
}

// ─── RESULT CARD ──────────────────────────────────────────────────────────────

class _ResultCard extends StatefulWidget {
  final RiskResult result;
  final VoidCallback onReset;
  final String? sessionId;

  const _ResultCard({required this.result, required this.onReset, this.sessionId});

  @override
  State<_ResultCard> createState() => _ResultCardState();
}

class _ResultCardState extends State<_ResultCard> {
  bool _isClinician = false;

  @override
  void initState() {
    super.initState();
    _checkMode();
  }
  
  Future<void> _checkMode() async {
    final prefs = await SharedPreferences.getInstance();
    if (mounted) {
      setState(() {
        _isClinician = prefs.getBool('clinician_mode') ?? false;
      });
    }
  }

  Color get _headerColor {
    switch (widget.result.level) {
      case RiskLevel.high:          return const Color(0xFF9B1C1C);
      case RiskLevel.caution:       return const Color(0xFFB45309);
      case RiskLevel.consult:       return const Color(0xFF1E5A8E);
      case RiskLevel.diffDiagnosis: return AppColors.sage;
      case RiskLevel.none:          return const Color(0xFF374151);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isMobile = Bp.isMobile(context);
    final fs = fontScale(context);

    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: AppColors.border),
        borderRadius: BorderRadius.circular(8),
        color: AppColors.white,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: double.infinity,
            padding: EdgeInsets.all(isMobile ? 18 : 24),
            decoration: BoxDecoration(
              color: _headerColor,
              borderRadius:
                  const BorderRadius.vertical(top: Radius.circular(8)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('ASSESSMENT RESULT',
                    style: GoogleFonts.sourceSans3(
                        fontSize: 10 * fs,
                        letterSpacing: 2.5,
                        color: Colors.white70)),
                const SizedBox(height: 6),
                Text(widget.result.title,
                    style: GoogleFonts.playfairDisplay(
                        fontSize: 20 * fs,
                        color: Colors.white,
                        fontWeight: FontWeight.w700)),
              ],
            ),
          ),
          Padding(
            padding: EdgeInsets.all(isMobile ? 16 : 24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(widget.result.body,
                    style: GoogleFonts.sourceSans3(
                        fontSize: 13 * fs, height: 1.8, color: AppColors.ink)),
                if (widget.result.diffNote != null) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: const BoxDecoration(
                      border:
                          Border(left: BorderSide(color: AppColors.sage, width: 3)),
                      color: Color(0xFFEEF6F3),
                    ),
                    child: Text(widget.result.diffNote!,
                        style: GoogleFonts.sourceSans3(
                            fontSize: 12 * fs,
                            color: AppColors.sage,
                            height: 1.6)),
                  ),
                ],
                const SizedBox(height: 14),
                ...widget.result.bullets.map((b) => Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Container(
                            width: 5,
                            height: 5,
                            margin: const EdgeInsets.only(top: 7, right: 10),
                            decoration: const BoxDecoration(
                                color: AppColors.rust, shape: BoxShape.circle),
                          ),
                          Expanded(
                              child: Text(b,
                                  style: GoogleFonts.sourceSans3(
                                      fontSize: 12 * fs,
                                      color: AppColors.muted,
                                      height: 1.7))),
                        ],
                      ),
                    )),
                const SizedBox(height: 14),
                SizedBox(
                  width: isMobile ? double.infinity : null,
                  child: OutlinedButton.icon(
                    onPressed: widget.onReset,
                    icon: const Icon(Icons.refresh, size: 16),
                    label: const Text('Start Over'),
                  ),
                ),
                if (_isClinician && widget.sessionId != null) ...[
                  const SizedBox(height: 16),
                  SizedBox(
                    width: isMobile ? double.infinity : null,
                    child: ElevatedButton.icon(
                      style: ElevatedButton.styleFrom(backgroundColor: AppColors.rust, foregroundColor: Colors.white),
                      onPressed: () => ClinicianDiagnosisBottomSheet.show(context, widget.sessionId!),
                      icon: const Icon(Icons.medical_services, size: 16),
                      label: const Text('Submit Clinical Diagnosis'),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}