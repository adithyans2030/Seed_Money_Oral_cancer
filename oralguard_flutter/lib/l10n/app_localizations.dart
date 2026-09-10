import 'package:flutter/material.dart';

class AppLocalizations {
  final Locale locale;

  AppLocalizations(this.locale);

  static AppLocalizations of(BuildContext context) {
    final instance = Localizations.of<AppLocalizations>(context, AppLocalizations);
    return instance ?? AppLocalizations(const Locale('en'));
  }

  static const LocalizationsDelegate<AppLocalizations> delegate =
      _AppLocalizationsDelegate();

  static final Map<String, Map<String, String>> _localizedValues = {
    'en': {
      'home': 'Home',
      'clinician_mode': 'Clinician Mode',
      'clinician_desc': 'Enable to log ground-truth clinical diagnoses',
      'hero_tag': '● ORAL CANCER SCREENING PLATFORM',
      'hero_t1': 'Detect early.\n',
      'hero_t2': 'Act sooner.\n',
      'hero_t3': 'Survive.',
      'hero_desc': 'Oral cancer is highly treatable when caught early. Use our tools to check your risk, review your symptoms, and compare lesion images against clinical cases.',
      'stat_5_1': '84%',
      'stat_5_1_lbl': '5-yr survival\nat Stage I',
      'stat_5_4': '20%',
      'stat_5_4_lbl': '5-yr survival\nat Stage IV',
      'self_exam': 'Self-Exam Guide',
      'self_exam_desc': 'Check your own mouth in 5 minutes',
      'screener': 'Risk Screener',
      'screener_desc': '18 questions — get a personalised risk result',
      'matcher': 'Image Matcher',
      'matcher_desc': 'Compare a lesion photo to our clinical database',
      'combined': 'Combined Triage',
      'combined_desc': 'Fuse questionnaire & image matcher for a final risk score',
      'start_here': 'START HERE',
      'open_guide': 'Open Guide →',
      'start_screener': 'Start Screener →',
      'open_matcher': 'Open Matcher →',
      'view_synthesis': 'View Synthesis →',
      'how_it_works': 'HOW IT WORKS',
      'three_tools': 'Three tools, one platform',
      'hiw_1_t': 'Learn to self-examine',
      'hiw_1_d': 'Follow the illustrated guide to inspect your mouth systematically.',
      'hiw_2_t': 'Answer the screener',
      'hiw_2_d': 'Our questionnaire weighs risk factors and symptoms to assess urgency.',
      'hiw_3_t': 'Match a lesion photo',
      'hiw_3_d': 'Upload an image and our AI finds the most visually similar cases.',
      'hiw_4_t': 'See a clinician',
      'hiw_4_d': 'These tools inform — they don\'t diagnose. Always consult a dentist or specialist.',
      'language': 'Language',
    },
    'hi': {
      'home': 'मुख्य पृष्ठ',
      'clinician_mode': 'क्लीनिक मोड',
      'clinician_desc': 'ग्राउंड-ट्रुथ क्लिनिकल डायग्नोसिस लॉग करने के लिए सक्षम करें',
      'hero_tag': '● ओरल कैंसर स्क्रीनिंग प्लेटफॉर्म',
      'hero_t1': 'जल्दी पता लगाएं।\n',
      'hero_t2': 'जल्दी कदम उठाएं।\n',
      'hero_t3': 'जीवित रहें।',
      'hero_desc': 'समय पर पता लगने पर मुंह के कैंसर का इलाज संभव है। अपने जोखिम की जांच करने, लक्षणों की समीक्षा करने और घाव की छवियों की तुलना करने के लिए हमारे टूल्स का उपयोग करें।',
      'stat_5_1': '84%',
      'stat_5_1_lbl': 'स्टेज I में\n5-वर्ष तक जीवित रहना',
      'stat_5_4': '20%',
      'stat_5_4_lbl': 'स्टेज IV में\n5-वर्ष तक जीवित रहना',
      'self_exam': 'सेल्फ-एग्जाम गाइड',
      'self_exam_desc': '5 मिनट में अपने मुंह की खुद जांच करें',
      'screener': 'रिस्क स्क्रीनर',
      'screener_desc': '18 प्रश्न — अपना व्यक्तिगत जोखिम परिणाम प्राप्त करें',
      'matcher': 'इमेज मैचर',
      'matcher_desc': 'घाव की तस्वीर की हमारे क्लिनिकल डेटाबेस से तुलना करें',
      'combined': 'कंबाइंड ट्राइएज',
      'combined_desc': 'अंतिम जोखिम स्कोर के लिए प्रश्नावली और इमेज मैचर को मिलाएं',
      'start_here': 'यहाँ से शुरू करें',
      'open_guide': 'गाइड खोलें →',
      'start_screener': 'स्क्रीनर शुरू करें →',
      'open_matcher': 'मैचर खोलें →',
      'view_synthesis': 'परिणाम देखें →',
      'how_it_works': 'यह कैसे काम करता है',
      'three_tools': 'तीन टूल्स, एक प्लेटफॉर्म',
      'hiw_1_t': 'सेल्फ-एग्जाम करना सीखें',
      'hiw_1_d': 'अपने मुंह का व्यवस्थित रूप से निरीक्षण करने के लिए सचित्र गाइड का पालन करें।',
      'hiw_2_t': 'स्क्रीनर का उत्तर दें',
      'hiw_2_d': 'हमारी प्रश्नावली जोखिम कारकों और लक्षणों का मूल्यांकन करती है।',
      'hiw_3_t': 'घाव की फोटो मैच करें',
      'hiw_3_d': 'एक छवि अपलोड करें और हमारा AI सबसे समान मामले ढूंढता है।',
      'hiw_4_t': 'डॉक्टर से मिलें',
      'hiw_4_d': 'ये टूल्स केवल जानकारी देते हैं, निदान नहीं। हमेशा डॉक्टर से सलाह लें।',
      'language': 'भाषा',
    },
    'kn': {
      'home': 'ಮುಖಪುಟ',
      'clinician_mode': 'ವೈದ್ಯರ ಮೋಡ್',
      'clinician_desc': 'ಕ್ಲಿನಿಕಲ್ ಡಯಾಗ್ನೋಸಿಸ್ ದಾಖಲಿಸಲು ಸಕ್ರಿಯಗೊಳಿಸಿ',
      'hero_tag': '● ಬಾಯಿ ಕ್ಯಾನ್ಸರ್ ತಪಾಸಣಾ ವೇದಿಕೆ',
      'hero_t1': 'ಬೇಗ ಪತ್ತೆಹಚ್ಚಿ.\n',
      'hero_t2': 'ಬೇಗ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ.\n',
      'hero_t3': 'ಬದುಕುಳಿಯಿರಿ.',
      'hero_desc': 'ಆರಂಭದಲ್ಲಿಯೇ ಪತ್ತೆಯಾದಾಗ ಬಾಯಿ ಕ್ಯಾನ್ಸರ್‌ಗೆ ಚಿಕಿತ್ಸೆ ಸಾಧ್ಯವಿದೆ. ನಿಮ್ಮ ಅಪಾಯವನ್ನು ಪರಿಶೀಲಿಸಲು ನಮ್ಮ ಸಾಧನಗಳನ್ನು ಬಳಸಿ.',
      'stat_5_1': '84%',
      'stat_5_1_lbl': 'ಹಂತ I ರಲ್ಲಿ\n5 ವರ್ಷದ ಉಳಿಯುವಿಕೆ',
      'stat_5_4': '20%',
      'stat_5_4_lbl': 'ಹಂತ IV ರಲ್ಲಿ\n5 ವರ್ಷದ ಉಳಿಯುವಿಕೆ',
      'self_exam': 'ಸ್ವಯಂ ತಪಾಸಣಾ ಮಾರ್ಗದರ್ಶಿ',
      'self_exam_desc': '5 ನಿಮಿಷಗಳಲ್ಲಿ ನಿಮ್ಮ ಬಾಯಿಯನ್ನು ಪರೀಕ್ಷಿಸಿ',
      'screener': 'ಅಪಾಯ ಸ್ಕ್ರೀನರ್',
      'screener_desc': '18 ಪ್ರಶ್ನೆಗಳು — ನಿಮ್ಮ ವೈಯಕ್ತಿಕ ಅಪಾಯದ ಫಲಿತಾಂಶವನ್ನು ಪಡೆಯಿರಿ',
      'matcher': 'ಚಿತ್ರ ಪರೀಕ್ಷಕ',
      'matcher_desc': 'ಗಾಯದ ಫೋಟೋವನ್ನು ನಮ್ಮ ಡೇಟಾಬೇಸ್‌ನೊಂದಿಗೆ ಹೋಲಿಸಿ',
      'combined': 'ಸಂಯೋಜಿತ ಟ್ರಯಾಜ್',
      'combined_desc': 'ಅಂತಿಮ ಅಪಾಯದ ಸ್ಕೋರ್‌ಗಾಗಿ ಎರಡನ್ನೂ ಸಂಯೋಜಿಸಿ',
      'start_here': 'ಇಲ್ಲಿ ಪ್ರಾರಂಭಿಸಿ',
      'open_guide': 'ಮಾರ್ಗದರ್ಶಿ ತೆರೆಯಿರಿ →',
      'start_screener': 'ಸ್ಕ್ರೀನರ್ ಪ್ರಾರಂಭಿಸಿ →',
      'open_matcher': 'ಮ್ಯಾಚರ್ ತೆರೆಯಿರಿ →',
      'view_synthesis': 'ಫಲಿತಾಂಶ ವೀಕ್ಷಿಸಿ →',
      'how_it_works': 'ಇದು ಹೇಗೆ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ',
      'three_tools': 'ಮೂರು ಸಾಧನಗಳು, ಒಂದು ವೇದಿಕೆ',
      'hiw_1_t': 'ಸ್ವಯಂ-ತಪಾಸಣೆಯನ್ನು ಕಲಿಯಿರಿ',
      'hiw_1_d': 'ನಿಮ್ಮ ಬಾಯಿಯನ್ನು ಪರೀಕ್ಷಿಸಲು ಸಚಿತ್ರ ಮಾರ್ಗದರ್ಶಿಯನ್ನು ಅನುಸರಿಸಿ.',
      'hiw_2_t': 'ಸ್ಕ್ರೀನರ್‌ಗೆ ಉತ್ತರಿಸಿ',
      'hiw_2_d': 'ಅಪಾಯದ ಅಂಶಗಳು ಮತ್ತು ರೋಗಲಕ್ಷಣಗಳನ್ನು ಮೌಲ್ಯಮಾಪನ ಮಾಡುತ್ತದೆ.',
      'hiw_3_t': 'ಚಿತ್ರವನ್ನು ಹೋಲಿಸಿ',
      'hiw_3_d': 'ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ, AI ಹೋಲಿಕೆಯ ಪ್ರಕರಣಗಳನ್ನು ಹುಡುಕುತ್ತದೆ.',
      'hiw_4_t': 'ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ',
      'hiw_4_d': 'ಯಾವಾಗಲೂ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.',
      'language': 'ಭಾಷೆ',
    }
  };

  String get(String key) {
    return _localizedValues[locale.languageCode]?[key] ?? _localizedValues['en']?[key] ?? key;
  }
}

class _AppLocalizationsDelegate
    extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();

  @override
  bool isSupported(Locale locale) => ['en', 'hi', 'kn'].contains(locale.languageCode);

  @override
  Future<AppLocalizations> load(Locale locale) async {
    return AppLocalizations(locale);
  }

  @override
  bool shouldReload(_AppLocalizationsDelegate old) => false;
}
