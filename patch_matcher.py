import re

def main():
    path = "oralguard_flutter/lib/screens/matcher_screen.dart"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # 1. Imports
    content = content.replace(
        "import '../widgets/nav_bar.dart';",
        "import '../widgets/nav_bar.dart';\nimport '../widgets/clinician_bottom_sheet.dart';\nimport 'package:shared_preferences/shared_preferences.dart';"
    )
    
    # 2. State variable
    content = content.replace(
        "List<MatchResult> _malignant = [];\n  final _picker = ImagePicker();",
        "List<MatchResult> _malignant = [];\n  String? _sessionId;\n  final _picker = ImagePicker();"
    )
    
    # 3. Save sessionId
    content = content.replace(
        """        setState(() {
          _benign = benignList;
          _malignant = malignantList;
          _showQuality = all.isNotEmpty && lowCount > all.length / 2;
          _loading = false;
        });""",
        """        setState(() {
          _sessionId = sessionId;
          _benign = benignList;
          _malignant = malignantList;
          _showQuality = all.isNotEmpty && lowCount > all.length / 2;
          _loading = false;
        });"""
    )
    
    # 4. Remove image reset
    content = content.replace(
        """      _malignant = [];
      _showQuality = false;
      _loading = false;""",
        """      _malignant = [];
      _sessionId = null;
      _showQuality = false;
      _loading = false;"""
    )
    
    # 5. Pass sessionId
    content = content.replace(
        """                if (_benign.isNotEmpty || _malignant.isNotEmpty)
                  _ResultsSection(
                    benign: _benign,
                    malignant: _malignant,
                    userImageBytes: _pickedBytes,
                    isWide: isWide,
                  ),""",
        """                if (_benign.isNotEmpty || _malignant.isNotEmpty)
                  _ResultsSection(
                    benign: _benign,
                    malignant: _malignant,
                    userImageBytes: _pickedBytes,
                    isWide: isWide,
                    sessionId: _sessionId,
                  ),"""
    )
    
    # 6. Make _ResultsSection Stateful
    content = content.replace(
        """class _ResultsSection extends StatelessWidget {
  final List<MatchResult> benign;
  final List<MatchResult> malignant;
  final Uint8List? userImageBytes;
  final bool isWide;

  const _ResultsSection({
    required this.benign,
    required this.malignant,
    required this.userImageBytes,
    required this.isWide,
  });

  @override
  Widget build(BuildContext context) {""",
        """class _ResultsSection extends StatefulWidget {
  final List<MatchResult> benign;
  final List<MatchResult> malignant;
  final Uint8List? userImageBytes;
  final bool isWide;
  final String? sessionId;

  const _ResultsSection({
    required this.benign,
    required this.malignant,
    required this.userImageBytes,
    required this.isWide,
    this.sessionId,
  });

  @override
  State<_ResultsSection> createState() => _ResultsSectionState();
}

class _ResultsSectionState extends State<_ResultsSection> {
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

  @override
  Widget build(BuildContext context) {"""
    )
    
    # 7. Add widget. prefixes
    content = content.replace("benign,", "widget.benign,")
    content = content.replace("malignant,", "widget.malignant,")
    content = content.replace("userImageBytes)", "widget.userImageBytes)")
    content = content.replace("isWide", "widget.isWide")
    content = content.replace("widget.widget.isWide", "widget.isWide") # fix double
    
    # 8. Add button at the end
    content = content.replace(
        """                  _ResultColumn(
                      results: widget.malignant,
                      type: 'malignant',
                      userImageBytes: widget.userImageBytes),
                ],
              ),
      ],
    );""",
        """                  _ResultColumn(
                      results: widget.malignant,
                      type: 'malignant',
                      userImageBytes: widget.userImageBytes),
                ],
              ),
        if (_isClinician && widget.sessionId != null) ...[
          const SizedBox(height: 24),
          SizedBox(
            width: widget.isWide ? null : double.infinity,
            child: ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.rust,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              ),
              onPressed: () => ClinicianDiagnosisBottomSheet.show(context, widget.sessionId!),
              icon: const Icon(Icons.medical_services, size: 18),
              label: const Text('Submit Clinical Diagnosis'),
            ),
          ),
        ],
      ],
    );"""
    )
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Done")

if __name__ == "__main__":
    main()
