import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'theme.dart';
import 'screens/home_screen.dart';
import 'screens/questionnaire_screen.dart';
import 'screens/matcher_screen.dart';
import 'screens/self_exam_screen.dart';
import 'screens/combined_risk_screen.dart';

import 'package:flutter_localizations/flutter_localizations.dart';
import 'l10n/app_localizations.dart';

final ValueNotifier<Locale> appLocale = ValueNotifier<Locale>(const Locale('en'));

void main() {
  runApp(const OralGuardApp());
}

final _router = GoRouter(
  routes: [
    GoRoute(path: '/',          builder: (_, __) => const HomeScreen()),
    GoRoute(path: '/screener',  builder: (_, __) => const QuestionnaireScreen()),
    GoRoute(path: '/matcher',   builder: (_, __) => const MatcherScreen()),
    GoRoute(path: '/self-exam', builder: (_, __) => const SelfExamScreen()),
    GoRoute(path: '/combined-risk', builder: (_, __) => const CombinedRiskScreen()),
  ],
);

class OralGuardApp extends StatelessWidget {
  const OralGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<Locale>(
      valueListenable: appLocale,
      builder: (context, locale, child) {
        return MaterialApp.router(
          title: 'OralGuard',
          debugShowCheckedModeBanner: false,
          theme: AppTheme.theme,
          routerConfig: _router,
          locale: locale,
          supportedLocales: const [
            Locale('en', ''),
            Locale('hi', ''),
            Locale('kn', ''),
          ],
          localizationsDelegates: const [
            AppLocalizations.delegate,
            GlobalMaterialLocalizations.delegate,
            GlobalWidgetsLocalizations.delegate,
            GlobalCupertinoLocalizations.delegate,
          ],
        );
      },
    );
  }
}