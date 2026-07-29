import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../theme.dart';

class ClinicianDiagnosisBottomSheet extends StatefulWidget {
  final String sessionId;

  const ClinicianDiagnosisBottomSheet({super.key, required this.sessionId});

  static void show(BuildContext context, String sessionId) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: AppColors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(12)),
      ),
      builder: (ctx) => Padding(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(ctx).viewInsets.bottom,
        ),
        child: ClinicianDiagnosisBottomSheet(sessionId: sessionId),
      ),
    );
  }

  @override
  State<ClinicianDiagnosisBottomSheet> createState() =>
      _ClinicianDiagnosisBottomSheetState();
}

class _ClinicianDiagnosisBottomSheetState
    extends State<ClinicianDiagnosisBottomSheet> {
  final _notesController = TextEditingController();
  final _clinicianIdController = TextEditingController();
  String? _diagnosis;
  bool _isSubmitting = false;

  final _options = ['Benign', 'Malignant', 'Needs Biopsy', 'Inconclusive'];

  @override
  void initState() {
    super.initState();
    _loadClinicianId();
  }

  Future<void> _loadClinicianId() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _clinicianIdController.text = prefs.getString('clinician_id') ?? '';
    });
  }

  Future<void> _submit() async {
    if (_diagnosis == null || _clinicianIdController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please provide diagnosis and clinician ID.')),
      );
      return;
    }

    setState(() => _isSubmitting = true);

    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('clinician_id', _clinicianIdController.text);

      final url = Uri.parse('https://gprabhanjana-oral-cancer-cbir-api.hf.space/clinician-label');
      final res = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          "session_id": widget.sessionId,
          "clinician_id": _clinicianIdController.text,
          "actual_diagnosis": _diagnosis,
          "notes": _notesController.text,
        }),
      );

      if (res.statusCode == 200) {
        if (!mounted) return;
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Diagnosis recorded. Thank you.'),
            backgroundColor: AppColors.sage,
          ),
        );
      } else {
        throw Exception('Server error: ${res.statusCode}');
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Submission failed: $e')),
      );
    } finally {
      if (mounted) setState(() => _isSubmitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisSize: CrossAxisSize.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'Submit Clinical Diagnosis',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: AppColors.ink,
                ),
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<String>(
            value: _diagnosis,
            decoration: const InputDecoration(
              labelText: 'Diagnosis',
              border: OutlineInputBorder(),
            ),
            items: _options
                .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                .toList(),
            onChanged: (val) => setState(() => _diagnosis = val),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _clinicianIdController,
            decoration: const InputDecoration(
              labelText: 'Clinician ID',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _notesController,
            decoration: const InputDecoration(
              labelText: 'Clinical Notes (Optional)',
              border: OutlineInputBorder(),
            ),
            maxLines: 3,
          ),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            height: 48,
            child: ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.rust,
                foregroundColor: Colors.white,
              ),
              onPressed: _isSubmitting ? null : _submit,
              child: _isSubmitting
                  ? const SizedBox(
                      width: 20, height: 20, child: CircularProgressIndicator())
                  : const Text('Submit Diagnosis'),
            ),
          ),
        ],
      ),
    );
  }
}
