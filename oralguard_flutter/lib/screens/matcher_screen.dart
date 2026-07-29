import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../theme.dart';
import '../widgets/nav_bar.dart';
import '../widgets/clinician_bottom_sheet.dart';
import 'package:shared_preferences/shared_preferences.dart';

// ─── Data models ──────────────────────────────────────────────────────────────

class MatchResult {
  final String imagePath;
  final String label;
  final double similarity;

  const MatchResult({
    required this.imagePath,
    required this.label,
    required this.similarity,
  });

  factory MatchResult.fromJson(Map<String, dynamic> json) => MatchResult(
        imagePath: json['image_path'] ?? '',
        label: json['label'] ?? '',
        similarity: (json['similarity'] as num).toDouble(),
      );
}

// ─── Screen ───────────────────────────────────────────────────────────────────

class MatcherScreen extends StatefulWidget {
  const MatcherScreen({super.key});

  @override
  State<MatcherScreen> createState() => _MatcherScreenState();
}

class _MatcherScreenState extends State<MatcherScreen> {
  XFile? _pickedFile;
  Uint8List? _pickedBytes;
  bool _loading = false;
  bool _showQuality = false;
  List<MatchResult> _benign = [];
  List<MatchResult> _malignant = [];
  String? _sessionId;
  final _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    final f = await _picker.pickImage(source: source, imageQuality: 85);
    if (f == null) return;
    final bytes = await f.readAsBytes();
    setState(() {
      _pickedFile = f;
      _pickedBytes = bytes;
      _benign = [];
      _malignant = [];
      _showQuality = false;
    });
    await _analyze(f);
  }

  Future<void> _analyze(XFile file) async {
    setState(() => _loading = true);
    try {
      final bytes = _pickedBytes ?? await file.readAsBytes();
      final req = http.MultipartRequest(
        'POST',
        Uri.parse('https://gprabhanjana-oral-cancer-cbir-api.hf.space/search'),
      );
      req.files.add(http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: file.name,
      ));
      final streamed = await req.send();
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final benignList = (data['results']['benign'] as List? ?? [])
            .take(6)
            .map((j) => MatchResult.fromJson(j))
            .toList();
        final malignantList = (data['results']['malignant'] as List? ?? [])
            .take(6)
            .map((j) => MatchResult.fromJson(j))
            .toList();

        final all = [...benignList, ...malignantList];
        final lowCount = all.where((r) => r.similarity < 0.55).length;

        setState(() {
          _sessionId = sessionId;
          _benign = benignList;
          _malignant = malignantList;
          _showQuality = all.isNotEmpty && lowCount > all.length / 2;
          _loading = false;
        });
      } else {
        throw Exception('Server error: ${response.statusCode}');
      }
    } catch (e) {
      setState(() => _loading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Error: $e'),
          backgroundColor: AppColors.rust,
          duration: const Duration(seconds: 6),
        ));
      }
    }
  }

  void _removeImage() {
    setState(() {
      _pickedFile = null;
      _pickedBytes = null;
      _benign = [];
      _malignant = [];
      _sessionId = null;
      _showQuality = false;
      _loading = false;
    });
  }

  String _formatSize(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }

  @override
  Widget build(BuildContext context) {
    final isMobile = Bp.isMobile(context);
    final widget.isWide = Bp.widget.isWide(context);
    final hPad = isMobile ? 16.0 : 20.0;
    final fs = fontScale(context);

    return Scaffold(
      appBar: const OralGuardNavBar(),
      bottomNavigationBar: isMobile ? const OralGuardBottomNav() : null,
      body: SingleChildScrollView(
        padding: EdgeInsets.all(hPad),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 1100),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header
                Text('● VISUAL COMPARISON TOOL',
                    style: GoogleFonts.sourceSans3(
                        fontSize: 9 * fs,
                        letterSpacing: 3.5,
                        color: AppColors.rust,
                        fontWeight: FontWeight.w600)),
                const SizedBox(height: 6),
                Text('AI Image Matcher',
                    style: GoogleFonts.playfairDisplay(fontSize: 28 * fs)),
                const SizedBox(height: 8),
                Text(
                  'Upload a photo of an oral lesion. The AI retrieves the top 6 most visually similar benign and malignant cases from our clinical image database.',
                  style: GoogleFonts.sourceSans3(
                      fontSize: 13 * fs,
                      color: AppColors.muted,
                      height: 1.75),
                ),
                const SizedBox(height: 24),

                // Upload
                _UploadSection(
                  pickedFile: _pickedFile,
                  pickedBytes: _pickedBytes,
                  loading: _loading,
                  onPickFile: () => _pickImage(ImageSource.gallery),
                  onOpenCamera: () => _pickImage(ImageSource.camera),
                  onRemove: _removeImage,
                  formatSize: _formatSize,
                  onOpenSelfExam: () => context.push('/self-exam'),
                  isMobile: isMobile,
                ),

                if (_showQuality) ...[
                  const SizedBox(height: 18),
                  _QualityWarning(),
                ],

                if (_benign.isNotEmpty || _malignant.isNotEmpty) ...[
                  const SizedBox(height: 28),
                  _ResultsSection(
                    benign: _widget.benign,
                    malignant: _widget.malignant,
                    userImageBytes: _pickedBytes,
                    widget.isWide: widget.isWide,
                  ),
                ],
                const SizedBox(height: 36),

                _MatcherFooter(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ─── UPLOAD SECTION ───────────────────────────────────────────────────────────

class _UploadSection extends StatelessWidget {
  final XFile? pickedFile;
  final Uint8List? pickedBytes;
  final bool loading;
  final VoidCallback onPickFile;
  final VoidCallback onOpenCamera;
  final VoidCallback onRemove;
  final String Function(int) formatSize;
  final VoidCallback onOpenSelfExam;
  final bool isMobile;

  const _UploadSection({
    required this.pickedFile,
    required this.pickedBytes,
    required this.loading,
    required this.onPickFile,
    required this.onOpenCamera,
    required this.onRemove,
    required this.formatSize,
    required this.onOpenSelfExam,
    required this.isMobile,
  });

  @override
  Widget build(BuildContext context) {
    final fs = fontScale(context);

    return Container(
      decoration: BoxDecoration(
        color: AppColors.white,
        border: Border.all(color: AppColors.border),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          // Header
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: const BoxDecoration(
              color: Color(0xFFFDFAF7),
              border: Border(bottom: BorderSide(color: AppColors.border)),
              borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
            ),
            child: Row(
              children: [
                Text('Upload Lesion Image',
                    style: GoogleFonts.playfairDisplay(fontSize: 15 * fs)),
                const Spacer(),
                isMobile
                    ? IconButton(
                        icon: const Icon(Icons.search, color: AppColors.rustMid),
                        tooltip: 'Self-Exam Guide',
                        onPressed: onOpenSelfExam,
                      )
                    : ElevatedButton.icon(
                        onPressed: onOpenSelfExam,
                        icon: const Icon(Icons.search, size: 16),
                        label: const Text('Self-Exam Guide'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.rustMid,
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 9),
                          textStyle: GoogleFonts.sourceSans3(
                              fontSize: 13, fontWeight: FontWeight.w600),
                        ),
                      ),
              ],
            ),
          ),

          // Body
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                isMobile
                    ? Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          ElevatedButton.icon(
                            onPressed: onPickFile,
                            icon: const Icon(Icons.folder_open, size: 18),
                            label: const Text('Upload File'),
                          ),
                          const SizedBox(height: 10),
                          OutlinedButton.icon(
                            onPressed: onOpenCamera,
                            icon: const Icon(Icons.camera_alt_outlined, size: 18),
                            label: const Text('Open Camera'),
                          ),
                        ],
                      )
                    : Row(
                        children: [
                          ElevatedButton.icon(
                            onPressed: onPickFile,
                            icon: const Icon(Icons.folder_open, size: 18),
                            label: const Text('Upload File'),
                          ),
                          const SizedBox(width: 10),
                          OutlinedButton.icon(
                            onPressed: onOpenCamera,
                            icon: const Icon(Icons.camera_alt_outlined, size: 18),
                            label: const Text('Open Camera'),
                          ),
                        ],
                      ),
                const SizedBox(height: 8),
                Text('Accepted: JPG, PNG, WEBP — max one image at a time',
                    style: GoogleFonts.sourceSans3(
                        fontSize: 11 * fs, color: AppColors.muted)),

                if (pickedFile != null && pickedBytes != null) ...[
                  const SizedBox(height: 16),
                  _ImagePreview(
                    file: pickedFile!,
                    bytes: pickedBytes!,
                    onRemove: onRemove,
                    formatSize: formatSize,
                  ),
                ],

                if (loading) ...[
                  const SizedBox(height: 14),
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: AppColors.rustLight,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: AppColors.rust.withOpacity(0.2)),
                    ),
                    child: Row(
                      children: [
                        const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                              strokeWidth: 2, color: AppColors.rust),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                              'Analyzing image against clinical database…',
                              style: GoogleFonts.sourceSans3(
                                  fontSize: 12 * fs,
                                  fontWeight: FontWeight.w600,
                                  color: AppColors.rust)),
                        ),
                      ],
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

class _ImagePreview extends StatelessWidget {
  final XFile file;
  final Uint8List bytes;
  final VoidCallback onRemove;
  final String Function(int) formatSize;

  const _ImagePreview({
    required this.file,
    required this.bytes,
    required this.onRemove,
    required this.formatSize,
  });

  @override
  Widget build(BuildContext context) {
    final isMobile = Bp.isMobile(context);
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: AppColors.border),
        borderRadius: BorderRadius.circular(12),
        color: const Color(0xFFFDFAF7),
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            child: Row(
              children: [
                const Icon(Icons.image_outlined, size: 20, color: AppColors.muted),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(file.name,
                      style: GoogleFonts.sourceSans3(
                          fontSize: 13, fontWeight: FontWeight.w600),
                      overflow: TextOverflow.ellipsis),
                ),
                IconButton(
                  onPressed: onRemove,
                  icon: const Icon(Icons.close, size: 16),
                  style: IconButton.styleFrom(
                    backgroundColor: AppColors.rust,
                    foregroundColor: Colors.white,
                    minimumSize: const Size(28, 28),
                    padding: EdgeInsets.zero,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(6)),
                  ),
                ),
              ],
            ),
          ),
          ClipRRect(
            borderRadius:
                const BorderRadius.vertical(bottom: Radius.circular(12)),
            child: Image.memory(
              bytes,
              height: isMobile ? 200 : 240,
              width: double.infinity,
              fit: BoxFit.contain,
            ),
          ),
        ],
      ),
    );
  }
}

// ─── QUALITY WARNING ──────────────────────────────────────────────────────────

class _QualityWarning extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final fs = fontScale(context);
    final tips = [
      'Ensure the area is well-lit — use a flashlight or bright natural light',
      'Make sure the lesion is clearly in the centre of the frame',
      'Hold the camera steady and close enough to see texture clearly',
      'Avoid blurry or out-of-focus images — check sharpness before uploading',
      'Remove any obstructions such as fingers, shadows, or dental tools',
      "Use the camera's macro or portrait mode if available",
    ];

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF8E6),
        border: Border.all(color: const Color(0xFFE8C84A)),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
              '⚠ Low Similarity Detected — Consider Retaking the Photo',
              style: GoogleFonts.sourceSans3(
                  fontSize: 13 * fs,
                  fontWeight: FontWeight.w600,
                  color: const Color(0xFF7A5C00))),
          const SizedBox(height: 8),
          Text(
              'More than half of the matched images have similarity scores below 55%, which may indicate the uploaded image is unclear.',
              style: GoogleFonts.sourceSans3(
                  fontSize: 12 * fs,
                  color: const Color(0xFF8A6A10),
                  height: 1.6)),
          const SizedBox(height: 10),
          ...tips.map((t) => Padding(
                padding: const EdgeInsets.only(bottom: 5),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('→ ',
                        style: GoogleFonts.sourceSans3(color: AppColors.amber)),
                    Expanded(
                        child: Text(t,
                            style: GoogleFonts.sourceSans3(
                                fontSize: 12 * fs,
                                color: const Color(0xFF8A6A10),
                                height: 1.5))),
                  ],
                ),
              )),
        ],
      ),
    );
  }
}

// ─── RESULTS SECTION ──────────────────────────────────────────────────────────

class _ResultsSection extends StatefulWidget {
  final List<MatchResult> benign;
  final List<MatchResult> malignant;
  final Uint8List? userImageBytes;
  final bool widget.isWide;
  final String? sessionId;

  const _ResultsSection({
    required this.widget.benign,
    required this.widget.malignant,
    required this.userImageBytes,
    required this.widget.isWide,
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
  Widget build(BuildContext context) {
    final fs = fontScale(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          crossAxisAlignment: WrapCrossAlignment.center,
          spacing: 10,
          runSpacing: 4,
          children: [
            Text('Matched Cases',
                style: GoogleFonts.playfairDisplay(fontSize: 20 * fs)),
            Text('Tap any image to compare with your photo',
                style: GoogleFonts.sourceSans3(
                    fontSize: 12 * fs, color: AppColors.muted)),
          ],
        ),
        const SizedBox(height: 16),
        widget.isWide
            ? Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                      child: _ResultColumn(
                          results: widget.benign,
                          type: 'benign',
                          userImageBytes: widget.userImageBytes)),
                  Container(
                    width: 1,
                    color: AppColors.border,
                    margin: const EdgeInsets.symmetric(horizontal: 20),
                    child: const SizedBox(height: 300),
                  ),
                  Expanded(
                      child: _ResultColumn(
                          results: widget.malignant,
                          type: 'malignant',
                          userImageBytes: widget.userImageBytes)),
                ],
              )
            : Column(
                children: [
                  _ResultColumn(
                      results: widget.benign,
                      type: 'benign',
                      userImageBytes: widget.userImageBytes),
                  const SizedBox(height: 24),
                  _ResultColumn(
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
    );
  }
}

class _ResultColumn extends StatelessWidget {
  final List<MatchResult> results;
  final String type;
  final Uint8List? userImageBytes;

  const _ResultColumn(
      {required this.results, required this.type, required this.userImageBytes});

  @override
  Widget build(BuildContext context) {
    final isBenign = type == 'benign';
    final color = isBenign ? AppColors.sage : AppColors.rust;
    final label = isBenign ? 'Benign Matches' : 'Malignant Matches';
    final fs = fontScale(context);

    final crossCount = Bp.widget.isWide(context) ? 3 : (Bp.isMobile(context) ? 2 : 3);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(children: [
          Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
          const SizedBox(width: 8),
          Text(label.toUpperCase(),
              style: GoogleFonts.sourceSans3(
                  fontSize: 9 * fs,
                  letterSpacing: 2.5,
                  color: color,
                  fontWeight: FontWeight.w600)),
        ]),
        const SizedBox(height: 10),
        GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: crossCount,
            mainAxisSpacing: 10,
            crossAxisSpacing: 10,
          ),
          itemCount: results.length,
          itemBuilder: (_, i) => _MatchCard(
            result: results[i],
            type: type,
            onTap: () => _openLightbox(context, results[i], type),
          ),
        ),
      ],
    );
  }

  void _openLightbox(BuildContext context, MatchResult result, String type) {
    showDialog(
      context: context,
      builder: (_) => _LightboxDialog(
          result: result, type: type, userImageBytes: widget.userImageBytes),
    );
  }
}

class _MatchCard extends StatefulWidget {
  final MatchResult result;
  final String type;
  final VoidCallback onTap;

  const _MatchCard(
      {required this.result, required this.type, required this.onTap});

  @override
  State<_MatchCard> createState() => _MatchCardState();
}

class _MatchCardState extends State<_MatchCard> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final color = widget.type == 'benign' ? AppColors.sage : AppColors.rust;
    final fs = fontScale(context);

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          transform: Matrix4.translationValues(0, _hovered ? -3 : 0, 0),
          decoration: BoxDecoration(
            color: AppColors.white,
            border: Border.all(color: _hovered ? color : AppColors.border),
            borderRadius: BorderRadius.circular(10),
            boxShadow: _hovered
                ? [
                    BoxShadow(
                        color: Colors.black.withOpacity(0.1),
                        blurRadius: 12,
                        offset: const Offset(0, 4))
                  ]
                : [],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: ClipRRect(
                  borderRadius:
                      const BorderRadius.vertical(top: Radius.circular(10)),
                  child: Image.network(
                    widget.result.imagePath,
                    width: double.infinity,
                    fit: BoxFit.cover,
                    errorBuilder: (_, __, ___) => Container(
                      color: AppColors.border,
                      child: const Icon(Icons.broken_image,
                          color: AppColors.muted),
                    ),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(6),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(widget.result.label,
                        style: GoogleFonts.sourceSans3(
                            fontSize: 9 * fs,
                            fontWeight: FontWeight.w600,
                            color: color,
                            letterSpacing: 0.5),
                        overflow: TextOverflow.ellipsis),
                    Text(
                        '${(widget.result.similarity * 100).toStringAsFixed(1)}% match',
                        style: GoogleFonts.sourceSans3(
                            fontSize: 9 * fs, color: AppColors.muted)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── LIGHTBOX ─────────────────────────────────────────────────────────────────

class _LightboxDialog extends StatelessWidget {
  final MatchResult result;
  final String type;
  final Uint8List? userImageBytes;

  const _LightboxDialog(
      {required this.result, required this.type, required this.userImageBytes});

  @override
  Widget build(BuildContext context) {
    final isBenign = type == 'benign';
    final color = isBenign ? AppColors.sage : AppColors.rust;
    final bgColor = isBenign ? AppColors.sageLight : AppColors.rustLight;
    final isMobile = Bp.isMobile(context);

    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      insetPadding: EdgeInsets.symmetric(
        horizontal: isMobile ? 12 : 40,
        vertical: isMobile ? 24 : 40,
      ),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 880),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Header
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: const BoxDecoration(
                color: Color(0xFFFDFAF7),
                border: Border(bottom: BorderSide(color: AppColors.border)),
                borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
              ),
              child: Row(
                children: [
                  Flexible(
                    child: Text('Visual Comparison',
                        style: GoogleFonts.playfairDisplay(fontSize: 15)),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
                    decoration: BoxDecoration(
                      color: bgColor,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      isBenign ? 'Benign' : 'Malignant',
                      style: GoogleFonts.sourceSans3(
                          fontSize: 10,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 2,
                          color: color),
                    ),
                  ),
                  const Spacer(),
                  IconButton(
                    onPressed: () => Navigator.pop(context),
                    icon: const Icon(Icons.close, size: 18, color: AppColors.muted),
                    style: IconButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(50),
                        side: const BorderSide(color: AppColors.border),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Body — side-by-side or stacked
            isMobile
                ? SingleChildScrollView(
                    child: Column(
                      children: [
                        _LightboxPanel(
                          label: 'Matched Case',
                          sublabel:
                              '${result.label} — ${(result.similarity * 100).toStringAsFixed(1)}% similarity',
                          child: Image.network(result.imagePath,
                              fit: BoxFit.contain,
                              errorBuilder: (_, __, ___) =>
                                  const Icon(Icons.broken_image)),
                        ),
                        const Divider(height: 1),
                        _LightboxPanel(
                          label: 'Your Image',
                          sublabel: 'Uploaded image',
                          child: userImageBytes != null
                              ? Image.memory(userImageBytes!, fit: BoxFit.contain)
                              : const Icon(Icons.image_not_supported,
                                  size: 48, color: AppColors.muted),
                        ),
                      ],
                    ),
                  )
                : IntrinsicHeight(
                    child: Row(
                      children: [
                        Expanded(
                            child: _LightboxPanel(
                          label: 'Matched Case',
                          sublabel:
                              '${result.label} — ${(result.similarity * 100).toStringAsFixed(1)}% similarity',
                          child: Image.network(result.imagePath,
                              fit: BoxFit.contain,
                              errorBuilder: (_, __, ___) =>
                                  const Icon(Icons.broken_image)),
                        )),
                        Container(width: 1, color: AppColors.border),
                        Expanded(
                            child: _LightboxPanel(
                          label: 'Your Image',
                          sublabel: 'Uploaded image',
                          child: userImageBytes != null
                              ? Image.memory(userImageBytes!, fit: BoxFit.contain)
                              : const Icon(Icons.image_not_supported,
                                  size: 48, color: AppColors.muted),
                        )),
                      ],
                    ),
                  ),
          ],
        ),
      ),
    );
  }
}

class _LightboxPanel extends StatelessWidget {
  final String label;
  final String sublabel;
  final Widget child;

  const _LightboxPanel(
      {required this.label, required this.sublabel, required this.child});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          Text(label.toUpperCase(),
              style: GoogleFonts.sourceSans3(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: AppColors.muted,
                  letterSpacing: 1.5)),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: Container(
              color: const Color(0xFFF5F1EB),
              width: double.infinity,
              height: 200,
              child: child,
            ),
          ),
          const SizedBox(height: 8),
          Text(sublabel,
              style:
                  GoogleFonts.sourceSans3(fontSize: 12, color: AppColors.muted),
              textAlign: TextAlign.center),
        ],
      ),
    );
  }
}

// ─── FOOTER ───────────────────────────────────────────────────────────────────

class _MatcherFooter extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(28),
      color: AppColors.dark,
      child: Column(
        children: [
          RichText(
              text: TextSpan(children: [
            TextSpan(
                text: 'Oral',
                style: GoogleFonts.playfairDisplay(
                    fontSize: 18, color: const Color(0xFFA09890))),
            TextSpan(
                text: 'Guard',
                style: GoogleFonts.playfairDisplay(
                    fontSize: 18, color: AppColors.rustMid)),
          ])),
          const SizedBox(height: 10),
          Text(
            'A research and educational screening platform. Not a substitute for professional clinical evaluation.',
            textAlign: TextAlign.center,
            style: GoogleFonts.sourceSans3(
                fontSize: 13, color: const Color(0xFF706860), height: 1.7),
          ),
          const SizedBox(height: 16),
          const Divider(color: Color(0xFF2A2520)),
          const SizedBox(height: 8),
          Text(
            '⚠ This tool does not provide medical diagnoses. Always consult a qualified dentist, oral surgeon, or oncologist.',
            textAlign: TextAlign.center,
            style: GoogleFonts.sourceSans3(
                fontSize: 12, color: const Color(0xFF4A4440)),
          ),
        ],
      ),
    );
  }
}