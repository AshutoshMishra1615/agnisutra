import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import 'profile_screen.dart';
import 'your_fields_screen.dart';
import 'alerts_screen.dart';
import 'ai_assistant_screen.dart';
import '../widgets/dashboard_widget.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final int _selectedIndex = 0;

  void _onItemTapped(int index) {
    if (index == 1) {
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => const YourFieldsScreen()),
      );
    } else if (index == 2) {
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => const ProfileScreen()),
      );
    }
  }

  void _showLanguageDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1E1E1E),
        title: Text(
          'select_language'.tr(),
          style: const TextStyle(color: Colors.white),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              title: Text(
                'english'.tr(),
                style: const TextStyle(color: Colors.white),
              ),
              onTap: () {
                context.setLocale(const Locale('en', 'US'));
                Navigator.pop(context);
              },
            ),
            ListTile(
              title: Text(
                'hindi'.tr(),
                style: const TextStyle(color: Colors.white),
              ),
              onTap: () {
                context.setLocale(const Locale('hi', 'IN'));
                Navigator.pop(context);
              },
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D0D0D),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0D0D0D), // Match dashboard background
        elevation: 0,
        title: Row(
          children: [
            Image.asset('assets/images/logo.png', height: 48, width: 48),
            const SizedBox(width: 8),
            const Text(
              'AgniSutra',
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 20,
              ),
            ),
          ],
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 8.0),
            child: ElevatedButton.icon(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const AiAssistantScreen(),
                  ),
                );
              },
              icon: const Icon(
                Icons.auto_awesome,
                size: 16,
                color: Colors.black,
              ),
              label: Text(
                'ai_assistant'.tr(),
                style: const TextStyle(
                  color: Colors.black,
                  fontWeight: FontWeight.bold,
                ),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFC5E1A5), // Light green
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 8,
                ),
              ),
            ),
          ),
          PopupMenuButton<String>(
            icon: const Icon(Icons.menu, color: Colors.white, size: 28),
            color: const Color(0xFF1E1E1E),
            onSelected: (value) {
              if (value == 'alerts') {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const AlertsScreen()),
                );
              } else if (value == 'language') {
                _showLanguageDialog();
              }
            },
            itemBuilder: (BuildContext context) => <PopupMenuEntry<String>>[
              PopupMenuItem<String>(
                value: 'alerts',
                child: Text(
                  'alerts'.tr(),
                  style: const TextStyle(color: Colors.white),
                ),
              ),
              PopupMenuItem<String>(
                value: 'language',
                child: Text(
                  'select_language'.tr(),
                  style: const TextStyle(color: Colors.white),
                ),
              ),
            ],
          ),
          const SizedBox(width: 6),
        ],
      ),
      body: const DashboardWidget(),
      bottomNavigationBar: NavigationBarTheme(
        data: NavigationBarThemeData(
          indicatorColor: const Color(0xFF3E4D3C),
          labelTextStyle: MaterialStateProperty.all(
            const TextStyle(color: Colors.white, fontSize: 12),
          ),
          iconTheme: MaterialStateProperty.resolveWith((states) {
            if (states.contains(MaterialState.selected)) {
              return const IconThemeData(color: Colors.white);
            }
            return const IconThemeData(color: Colors.grey);
          }),
        ),
        child: NavigationBar(
          backgroundColor: const Color(0xFF1E1E1E),
          selectedIndex: _selectedIndex,
          onDestinationSelected: _onItemTapped,
          destinations: [
            NavigationDestination(
              icon: const Icon(Icons.spa_outlined),
              selectedIcon: const Icon(Icons.spa),
              label: 'crops'.tr(),
            ),
            NavigationDestination(
              icon: const Icon(Icons.map_outlined),
              selectedIcon: const Icon(Icons.map),
              label: 'land'.tr(),
            ),
            NavigationDestination(
              icon: const Icon(Icons.person_outline),
              selectedIcon: const Icon(Icons.person),
              label: 'profile'.tr(),
            ),
          ],
        ),
      ),
    );
  }
}
