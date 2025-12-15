import 'package:flutter/material.dart';
import 'language_selection_screen.dart';
import '../services/auth_service.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _initApp();
  }

  Future<void> _initApp() async {
    final authService = AuthService();

    // Run the timer and the data fetching in parallel
    // We enforce a timeout on the profile fetching so the splash screen doesn't hang forever
    // The splash screen will show for at least 2 seconds, and at most 5 seconds
    await Future.wait([
      Future.delayed(const Duration(seconds: 2)), // Minimum splash duration
      _preFetchUserProfile(authService).timeout(
        const Duration(seconds: 5),
        onTimeout: () {
          debugPrint("Profile pre-fetch timed out");
        },
      ),
    ]);

    if (mounted) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => const LanguageSelectionScreen(),
        ),
      );
    }
  }

  Future<void> _preFetchUserProfile(AuthService authService) async {
    try {
      // Check if we have a token
      final token = await authService.getToken();
      if (token != null) {
        // Pre-fetch profile so it's cached in AuthService
        // This makes the ProfileScreen load instantly later
        await authService.getUserProfile();
      }
    } catch (e) {
      debugPrint("Error pre-fetching profile: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1E1E1E),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Placeholder for Logo
            Icon(Icons.agriculture, size: 100, color: Colors.green),
            SizedBox(height: 20),
            Text(
              'AgniSutra',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            SizedBox(height: 20),
            CircularProgressIndicator(color: Colors.green),
          ],
        ),
      ),
    );
  }
}
