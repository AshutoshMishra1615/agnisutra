import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import 'home_screen.dart';

class LanguageSelectionScreen extends StatefulWidget {
  const LanguageSelectionScreen({super.key});

  @override
  State<LanguageSelectionScreen> createState() =>
      _LanguageSelectionScreenState();
}

class _LanguageSelectionScreenState extends State<LanguageSelectionScreen> {
  String _selectedLanguage = 'en'; // Default to English

  @override
  Widget build(BuildContext context) {
    return Scaffold(
<<<<<<< HEAD
      backgroundColor: const Color(0xFF1E1E1E), // Dark background
=======
      backgroundColor: const Color(0xFF010101), // Dark background
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const SizedBox(height: 40),
              // Logo and Title
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Image.asset('assets/images/logo.png', height: 40),
                  const SizedBox(width: 12),
                  const Text(
                    'AgniSutra',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              const Text(
                'Select your language',
                style: TextStyle(color: Colors.white70, fontSize: 16),
              ),
              const SizedBox(height: 40),

              // Language Options
              _buildLanguageOption('English', 'en'),
              const SizedBox(height: 16),
              _buildLanguageOption('Hindi', 'hi'),

              const Spacer(),

              // Accept Button
              SizedBox(
                width: double.infinity,
                height: 50,
                child: ElevatedButton(
                  onPressed: () {
                    // Set locale
                    if (_selectedLanguage == 'en') {
                      context.setLocale(const Locale('en', 'US'));
                    } else {
                      context.setLocale(const Locale('hi', 'IN'));
                    }

                    // Navigate to Home
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const HomeScreen(),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
<<<<<<< HEAD
                    backgroundColor: const Color(0xFFB0B0B0), // Greyish button
=======
                    backgroundColor: const Color(0xFF3D5F3E), // Green button
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(25),
                    ),
                  ),
                  child: const Text(
                    'Accept',
                    style: TextStyle(
<<<<<<< HEAD
                      color: Colors.black87,
=======
                      color: Colors.white,
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'I read and accept the Terms of use and the private policy',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.blue, fontSize: 12),
              ),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLanguageOption(String label, String code) {
    final isSelected = _selectedLanguage == code;
    return GestureDetector(
      onTap: () {
        setState(() {
          _selectedLanguage = code;
        });
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        decoration: BoxDecoration(
<<<<<<< HEAD
          color: const Color(0xFF333333),
          borderRadius: BorderRadius.circular(12),
          border: isSelected ? Border.all(color: Colors.white, width: 1) : null,
=======
          color: isSelected ? const Color(0xFF3D5F3E) : const Color(0xFF333333),
          borderRadius: BorderRadius.circular(12),
          border: isSelected
              ? Border.all(color: const Color(0xFF3D5F3E), width: 1)
              : null,
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w500,
              ),
            ),
<<<<<<< HEAD
            Container(
              width: 24,
              height: 24,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: isSelected ? Colors.white : Colors.grey,
                border: Border.all(color: Colors.grey),
              ),
            ),
=======
            if (isSelected)
              Container(
                width: 24,
                height: 24,
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white,
                ),
                child: const Icon(
                  Icons.check,
                  color: Color(0xFF3D5F3E),
                  size: 16,
                ),
              )
            else
              Container(
                width: 24,
                height: 24,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.grey),
                ),
              ),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
          ],
        ),
      ),
    );
  }
}
