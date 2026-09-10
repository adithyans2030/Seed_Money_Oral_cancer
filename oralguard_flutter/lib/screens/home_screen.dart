import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/nav_bar.dart';
import '../l10n/app_localizations.dart';
import '../main.dart';

import 'package:shared_preferences/shared_preferences.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isClinician = false;

  @override
  void initState() {
    super.initState();
    _loadMode();
  }

  Future<void> _loadMode() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _isClinician = prefs.getBool('clinician_mode') ?? false;
    });
  }

  Future<void> _toggleMode(bool val) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('clinician_mode', val);
    setState(() {
      _isClinician = val;
    });
  }

  @override
  Widget build(BuildContext context) {
    final isMobile = Bp.isMobile(context);

    return Scaffold(
      appBar: const OralGuardNavBar(),
      bottomNavigationBar: isMobile ? const OralGuardBottomNav() : null,
      body: SingleChildScrollView(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  const Icon(Icons.language, size: 18, color: AppColors.muted),
                  const SizedBox(width: 8),
                  DropdownButton<String>(
                    value: appLocale.value.languageCode,
                    underline: const SizedBox(),
                    items: const [
                      DropdownMenuItem(value: 'en', child: Text('English')),
                      DropdownMenuItem(value: 'hi', child: Text('हिंदी')),
                      DropdownMenuItem(value: 'kn', child: Text('ಕನ್ನಡ')),
                    ],
                    onChanged: (val) {
                      if (val != null) {
                        appLocale.value = Locale(val);
                      }
                    },
                  ),
                ],
              ),
            ),
            SwitchListTile(
              title: Text(AppLocalizations.of(context).get('clinician_mode')),
              subtitle: Text(AppLocalizations.of(context).get('clinician_desc')),
              value: _isClinician,
              onChanged: _toggleMode,
              activeColor: AppColors.rust,
            ),
            _HeroSection(),
            _FeaturesSection(),
            _HowItWorksSection(),
            _Footer(),
          ],
        ),
      ),
    );
  }
}

// ─── HERO ─────────────────────────────────────────────────────────────────────

class _HeroSection extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final isWide = Bp.isWide(context);
    final hPad = Bp.isMobile(context) ? 16.0 : 24.0;

    return Container(
      color: AppColors.dark,
      width: double.infinity,
      padding: EdgeInsets.symmetric(horizontal: hPad, vertical: isWide ? 56 : 36),
      child: isWide
          ? Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(child: _HeroText()),
                const SizedBox(width: 48),
                Expanded(child: _HeroCards()),
              ],
            )
          : Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _HeroText(),
                const SizedBox(height: 28),
                _HeroCards(),
              ],
            ),
    );
  }
}

class _HeroText extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final fs = fontScale(context);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
          decoration: BoxDecoration(
            border: Border.all(color: AppColors.rustMid.withOpacity(0.4)),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            AppLocalizations.of(context).get('hero_tag'),
            style: GoogleFonts.sourceSans3(
              fontSize: 9 * fs,
              letterSpacing: 2.5,
              color: AppColors.rustMid,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        const SizedBox(height: 14),
        RichText(
          text: TextSpan(
            children: [
              TextSpan(
                text: AppLocalizations.of(context).get('hero_t1'),
                style: GoogleFonts.playfairDisplay(
                  fontSize: 34 * fs,
                  color: const Color(0xFFF0ECE6),
                  height: 1.15,
                ),
              ),
              TextSpan(
                text: AppLocalizations.of(context).get('hero_t2'),
                style: GoogleFonts.playfairDisplay(
                  fontSize: 34 * fs,
                  color: AppColors.rustMid,
                  fontStyle: FontStyle.italic,
                  height: 1.15,
                ),
              ),
              TextSpan(
                text: AppLocalizations.of(context).get('hero_t3'),
                style: GoogleFonts.playfairDisplay(
                  fontSize: 34 * fs,
                  color: const Color(0xFFF0ECE6),
                  height: 1.15,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        Text(
          AppLocalizations.of(context).get('hero_desc'),
          style: GoogleFonts.sourceSans3(
            fontSize: 14 * fs,
            color: const Color(0xFFA09890),
            height: 1.8,
          ),
        ),
        const SizedBox(height: 24),
        Row(
          children: [
            _StatItem(number: AppLocalizations.of(context).get('stat_5_1'), label: AppLocalizations.of(context).get('stat_5_1_lbl')),
            const SizedBox(width: 36),
            _StatItem(number: AppLocalizations.of(context).get('stat_5_4'), label: AppLocalizations.of(context).get('stat_5_4_lbl')),
          ],
        ),
      ],
    );
  }
}

class _StatItem extends StatelessWidget {
  final String number;
  final String label;
  const _StatItem({required this.number, required this.label});

  @override
  Widget build(BuildContext context) {
    final fs = fontScale(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(number,
            style: GoogleFonts.playfairDisplay(fontSize: 26 * fs, color: AppColors.rustMid)),
        Text(label,
            style: GoogleFonts.sourceSans3(
                fontSize: 11 * fs, color: const Color(0xFF706860), height: 1.5)),
      ],
    );
  }
}

class _HeroCards extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final loc = AppLocalizations.of(context);
    return Column(
      children: [
        _HeroCard(
          icon: '🔍',
          iconBg: AppColors.rust.withOpacity(0.2),
          title: loc.get('self_exam'),
          desc: loc.get('self_exam_desc'),
          onTap: () => context.go('/self-exam'),
        ),
        const SizedBox(height: 10),
        _HeroCard(
          icon: '📋',
          iconBg: AppColors.sage.withOpacity(0.2),
          title: loc.get('screener'),
          desc: loc.get('screener_desc'),
          onTap: () => context.go('/screener'),
        ),
        const SizedBox(height: 10),
        _HeroCard(
          icon: '📷',
          iconBg: AppColors.rust.withOpacity(0.2),
          title: loc.get('matcher'),
          desc: loc.get('matcher_desc'),
          onTap: () => context.go('/matcher'),
        ),
        const SizedBox(height: 10),
        _HeroCard(
          icon: '🔬',
          iconBg: AppColors.rustMid.withOpacity(0.2),
          title: loc.get('combined'),
          desc: loc.get('combined_desc'),
          onTap: () => context.go('/combined-risk'),
        ),
      ],
    );
  }
}

class _HeroCard extends StatefulWidget {
  final String icon;
  final Color iconBg;
  final String title;
  final String desc;
  final VoidCallback onTap;

  const _HeroCard({
    required this.icon,
    required this.iconBg,
    required this.title,
    required this.desc,
    required this.onTap,
  });

  @override
  State<_HeroCard> createState() => _HeroCardState();
}

class _HeroCardState extends State<_HeroCard> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_)  => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.all(14),
          transform: Matrix4.translationValues(_hovered ? 4 : 0, 0, 0),
          decoration: BoxDecoration(
            color: _hovered
                ? Colors.white.withOpacity(0.10)
                : Colors.white.withOpacity(0.06),
            border: Border.all(
              color: _hovered
                  ? AppColors.rust.withOpacity(0.5)
                  : Colors.white.withOpacity(0.10),
            ),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: widget.iconBg,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Center(
                    child: Text(widget.icon, style: const TextStyle(fontSize: 19))),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(widget.title,
                        style: GoogleFonts.sourceSans3(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: const Color(0xFFF0ECE6))),
                    const SizedBox(height: 2),
                    Text(widget.desc,
                        style: GoogleFonts.sourceSans3(
                            fontSize: 12, color: const Color(0xFF706860))),
                  ],
                ),
              ),
              Icon(
                Icons.arrow_forward,
                color: _hovered ? AppColors.rustMid : const Color(0xFF706860),
                size: 18,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── FEATURES ─────────────────────────────────────────────────────────────────

class _FeaturesSection extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final isWide = Bp.isWide(context);
    final hPad = Bp.isMobile(context) ? 16.0 : 24.0;
    final loc = AppLocalizations.of(context);

    final cards = [
      _FeatureData(
        badge: loc.get('start_here'),
        title: loc.get('self_exam'),
        desc: loc.get('hiw_1_d'),
        btnLabel: loc.get('open_guide'),
        onTap: () => context.go('/self-exam'),
      ),
      _FeatureData(
        badge: 'RISK ASSESSMENT',
        title: loc.get('screener'),
        desc: loc.get('hiw_2_d'),
        btnLabel: loc.get('start_screener'),
        onTap: () => context.go('/screener'),
      ),
      _FeatureData(
        badge: 'VISUAL COMPARISON',
        title: loc.get('matcher'),
        desc: loc.get('hiw_3_d'),
        btnLabel: loc.get('open_matcher'),
        onTap: () => context.go('/matcher'),
      ),
      _FeatureData(
        badge: 'SYNTHESIS',
        title: loc.get('combined'),
        desc: loc.get('combined_desc'),
        btnLabel: loc.get('view_synthesis'),
        onTap: () => context.go('/combined-risk'),
        badgeColor: AppColors.rustMid,
      ),
      _FeatureData(
        badge: 'ALWAYS REMEMBER',
        title: loc.get('hiw_4_t'),
        desc: loc.get('hiw_4_d'),
        btnLabel: null,
        onTap: null,
        badgeColor: AppColors.sage,
      ),
    ];

    return Padding(
      padding: EdgeInsets.all(hPad),
      child: isWide
          ? GridView.count(
              crossAxisCount: 2,
              shrinkWrap: true,
              crossAxisSpacing: 16,
              mainAxisSpacing: 16,
              childAspectRatio: 1.4,
              physics: const NeverScrollableScrollPhysics(),
              children: cards.map((d) => _FeatureCard(data: d)).toList(),
            )
          : Column(
              children: cards
                  .map((d) => Padding(
                        padding: const EdgeInsets.only(bottom: 14),
                        child: _FeatureCard(data: d),
                      ))
                  .toList(),
            ),
    );
  }
}

class _FeatureData {
  final String badge;
  final String title;
  final String desc;
  final String? btnLabel;
  final VoidCallback? onTap;
  final Color badgeColor;

  const _FeatureData({
    required this.badge,
    required this.title,
    required this.desc,
    required this.btnLabel,
    required this.onTap,
    this.badgeColor = AppColors.rust,
  });
}

class _FeatureCard extends StatefulWidget {
  final _FeatureData data;
  const _FeatureCard({required this.data});

  @override
  State<_FeatureCard> createState() => _FeatureCardState();
}

class _FeatureCardState extends State<_FeatureCard> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final isMobile = Bp.isMobile(context);
    final fs = fontScale(context);

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_)  => setState(() => _hovered = false),
      child: GestureDetector(
        onTap: widget.data.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: EdgeInsets.all(isMobile ? 20 : 28),
          transform: Matrix4.translationValues(0, _hovered ? -2 : 0, 0),
          decoration: BoxDecoration(
            color: AppColors.white,
            border: Border.all(
              color: _hovered ? AppColors.rust : AppColors.border,
            ),
            borderRadius: BorderRadius.circular(12),
            boxShadow: _hovered
                ? [
                    BoxShadow(
                        color: AppColors.rust.withOpacity(0.10),
                        blurRadius: 24,
                        offset: const Offset(0, 8))
                  ]
                : [],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                widget.data.badge,
                style: GoogleFonts.sourceSans3(
                  fontSize: 9 * fs,
                  letterSpacing: 2.5,
                  color: widget.data.badgeColor,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 10),
              Text(widget.data.title,
                  style: GoogleFonts.playfairDisplay(fontSize: 20 * fs)),
              const SizedBox(height: 8),
              Text(
                widget.data.desc,
                style: GoogleFonts.sourceSans3(
                    fontSize: 13 * fs, color: AppColors.muted, height: 1.75),
              ),
              if (widget.data.btnLabel != null) ...[
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: widget.data.onTap,
                  child: Text(widget.data.btnLabel!.toUpperCase()),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

// ─── HOW IT WORKS ─────────────────────────────────────────────────────────────

class _HowItWorksSection extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final loc = AppLocalizations.of(context);
    final steps = [
      ('01', loc.get('hiw_1_t'), loc.get('hiw_1_d')),
      ('02', loc.get('hiw_2_t'), loc.get('hiw_2_d')),
      ('03', loc.get('hiw_3_t'), loc.get('hiw_3_d')),
      ('04', loc.get('hiw_4_t'), loc.get('hiw_4_d')),
    ];

    final isWide = Bp.isWide(context);
    final isMobile = Bp.isMobile(context);
    final hPad = isMobile ? 20.0 : 40.0;
    final fs = fontScale(context);

    return Container(
      color: const Color(0xFFFDFAF7),
      width: double.infinity,
      padding: EdgeInsets.symmetric(horizontal: hPad, vertical: 36),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(loc.get('how_it_works'),
              style: GoogleFonts.sourceSans3(
                  fontSize: 9 * fs,
                  letterSpacing: 3.5,
                  color: AppColors.rust,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          Text(loc.get('three_tools'),
              style: GoogleFonts.playfairDisplay(fontSize: 24 * fs)),
          const SizedBox(height: 24),
          isWide
              ? Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: steps
                      .map((s) => Expanded(
                          child: _StepTile(num: s.$1, title: s.$2, desc: s.$3)))
                      .toList(),
                )
              : GridView.count(
                  crossAxisCount: 2,
                  shrinkWrap: true,
                  crossAxisSpacing: 14,
                  mainAxisSpacing: 14,
                  childAspectRatio: isMobile ? 1.0 : 1.15,
                  physics: const NeverScrollableScrollPhysics(),
                  children: steps
                      .map((s) => _StepTile(num: s.$1, title: s.$2, desc: s.$3))
                      .toList(),
                ),
        ],
      ),
    );
  }
}

class _StepTile extends StatelessWidget {
  final String num;
  final String title;
  final String desc;
  const _StepTile({required this.num, required this.title, required this.desc});

  @override
  Widget build(BuildContext context) {
    final fs = fontScale(context);
    return Padding(
      padding: const EdgeInsets.only(right: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(num,
              style: GoogleFonts.playfairDisplay(
                  fontSize: 40 * fs, color: AppColors.rustLight)),
          const SizedBox(height: 6),
          Text(title,
              style: GoogleFonts.sourceSans3(
                  fontSize: 13 * fs, fontWeight: FontWeight.w600)),
          const SizedBox(height: 4),
          Text(desc,
              style: GoogleFonts.sourceSans3(
                  fontSize: 12 * fs, color: AppColors.muted, height: 1.65)),
        ],
      ),
    );
  }
}

// ─── FOOTER ───────────────────────────────────────────────────────────────────

class _Footer extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final hPad = Bp.isMobile(context) ? 20.0 : 40.0;

    return Container(
      color: AppColors.dark,
      width: double.infinity,
      padding: EdgeInsets.symmetric(horizontal: hPad, vertical: 32),
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
            ]),
          ),
          const SizedBox(height: 10),
          Text(
            'A research and educational screening platform. Not a substitute for professional clinical evaluation.',
            textAlign: TextAlign.center,
            style: GoogleFonts.sourceSans3(
                fontSize: 13, color: const Color(0xFF706860), height: 1.7),
          ),
          const SizedBox(height: 18),
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