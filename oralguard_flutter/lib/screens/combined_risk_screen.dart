import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../theme.dart';
import '../widgets/nav_bar.dart';

class CombinedRiskScreen extends StatefulWidget {
  const CombinedRiskScreen({super.key});

  @override
  State<CombinedRiskScreen> createState() => _CombinedRiskScreenState();
}

class _CombinedRiskScreenState extends State<CombinedRiskScreen> {
  bool _isLoading = true;
  String? _errorMsg;
  Map<String, dynamic>? _fusionData;

  @override
  void initState() {
    super.initState();
    _fetchCombinedRisk();
  }

  Future<void> _fetchCombinedRisk() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final sessionId = prefs.getString('cbir_session_id');

      if (sessionId == null || sessionId.isEmpty) {
        setState(() {
          _errorMsg = "No active session found. Please complete the Risk Screener and Image Matcher first.";
          _isLoading = false;
        });
        return;
      }

      final url = Uri.parse('https://gprabhanjana-oral-cancer-cbir-api.hf.space/combined-risk');
      final res = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({"session_id": sessionId}),
      );

      if (res.statusCode == 200) {
        setState(() {
          _fusionData = jsonDecode(res.body);
          _isLoading = false;
        });
      } else if (res.statusCode == 422) {
        setState(() {
          _errorMsg = "Session is incomplete. Both the Risk Screener and Image Matcher must be completed before a combined result can be generated.";
          _isLoading = false;
        });
      } else if (res.statusCode == 404) {
        setState(() {
          _errorMsg = "Session not found on server. Please start a new session.";
          _isLoading = false;
        });
      } else {
        setState(() {
          _errorMsg = "Server error (${res.statusCode}). Please try again later.";
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMsg = "Network error: $e";
        _isLoading = false;
      });
    }
  }

  Color _getUrgencyColor(String? urgency) {
    if (urgency == null) return AppColors.muted;
    final lower = urgency.toLowerCase();
    if (lower.contains('urgent') || lower.contains('high')) return AppColors.rust;
    if (lower.contains('monitor') || lower.contains('moderate')) return AppColors.rustMid;
    if (lower.contains('routine') || lower.contains('low')) return AppColors.sage;
    return AppColors.muted;
  }

  @override
  Widget build(BuildContext context) {
    final isMobile = Bp.isMobile(context);

    return Scaffold(
      appBar: const OralGuardNavBar(),
      bottomNavigationBar: isMobile ? const OralGuardBottomNav() : null,
      body: _isLoading
          ? const Center(child: CircularProgressIndicator(color: AppColors.rust))
          : _errorMsg != null
              ? _buildErrorState()
              : _buildResultState(),
    );
  }

  Widget _buildErrorState() {
    final fs = fontScale(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Container(
          constraints: const BoxConstraints(maxWidth: 600),
          padding: const EdgeInsets.all(32),
          decoration: BoxDecoration(
            color: const Color(0xFFFDFAF7),
            border: Border.all(color: AppColors.border),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.warning_amber_rounded, size: 48, color: AppColors.rustMid),
              const SizedBox(height: 16),
              Text(
                'Incomplete Triage',
                style: GoogleFonts.playfairDisplay(fontSize: 24 * fs, color: AppColors.ink),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),
              Text(
                _errorMsg ?? 'Unknown error',
                style: GoogleFonts.sourceSans3(fontSize: 14 * fs, color: AppColors.muted, height: 1.6),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildResultState() {
    final fs = fontScale(context);
    final urgency = _fusionData?['urgency'] ?? 'Unknown';
    final label = _fusionData?['combined_risk_label'] ?? 'Unknown Risk';
    final rec = _fusionData?['recommendation'] ?? 'Please consult a clinician.';
    final uColor = _getUrgencyColor(urgency);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24.0),
      child: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 800),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Combined Triage Result',
                style: GoogleFonts.playfairDisplay(fontSize: 32 * fs, color: AppColors.ink),
              ),
              const SizedBox(height: 8),
              Text(
                'This synthesis fuses your questionnaire responses and image analysis for a comprehensive assessment.',
                style: GoogleFonts.sourceSans3(fontSize: 14 * fs, color: AppColors.muted, height: 1.5),
              ),
              const SizedBox(height: 32),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(32),
                decoration: BoxDecoration(
                  color: uColor.withOpacity(0.05),
                  border: Border.all(color: uColor.withOpacity(0.5)),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: uColor,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        urgency.toUpperCase(),
                        style: GoogleFonts.sourceSans3(
                          fontSize: 12 * fs,
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1.5,
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      label,
                      style: GoogleFonts.playfairDisplay(fontSize: 26 * fs, color: AppColors.ink, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      rec,
                      style: GoogleFonts.sourceSans3(fontSize: 15 * fs, color: AppColors.ink, height: 1.6),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: const Color(0xFFF6F4F0),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AppColors.border),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.info_outline, color: AppColors.muted, size: 24),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Text(
                        'This result is generated by an AI assistant for research purposes. It is not a definitive medical diagnosis. Always rely on a qualified oral health specialist for final clinical judgment.',
                        style: GoogleFonts.sourceSans3(fontSize: 13 * fs, color: AppColors.muted, height: 1.5),
                      ),
                    ),
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
