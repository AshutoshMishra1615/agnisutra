import 'dart:ui';
import 'package:flutter/material.dart';
<<<<<<< HEAD
=======
import 'package:easy_localization/easy_localization.dart';
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
// import 'package:flutter_svg/flutter_svg.dart';
import 'home_screen.dart';
import 'register_screen.dart';
import '../services/auth_service.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _authService = AuthService();
  bool _isLoading = false;

  void _login() async {
    setState(() {
      _isLoading = true;
    });

    if (!_emailController.text.contains(
      RegExp(
        r"^[a-zA-Z0-9.a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9]+\.[a-zA-Z]+",
      ),
    )) {
<<<<<<< HEAD
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a valid email')),
      );
=======
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('enter_valid_email'.tr())));
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
    }

    // if (_emailController.text.isEmpty) {
    //   ScaffoldMessenger.of(
    //     context,
    //   ).showSnackBar(const SnackBar(content: Text('Please enter your name')));
    // }
    if (_passwordController.text.isEmpty) {
<<<<<<< HEAD
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter your password')),
      );
=======
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('enter_password'.tr())));
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
    }

    bool success = await _authService.login(
      _emailController.text,
      _passwordController.text,
    );

    setState(() {
      _isLoading = false;
    });

    if (success) {
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (context) => const HomeScreen()),
      );
    } else {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
<<<<<<< HEAD
      ).showSnackBar(const SnackBar(content: Text('Login Failed')));
=======
      ).showSnackBar(SnackBar(content: Text('login_failed'.tr())));
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.white.withOpacity(0.1),
        elevation: 1,
        automaticallyImplyLeading: false,
        flexibleSpace: ClipRRect(
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: Container(color: Colors.transparent),
          ),
        ),
        title: Row(
          children: [
            Image.asset('assets/images/logo.png', height: 60, width: 60),
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
      ),
      body: Stack(
        children: [
          // Background Image
          Container(
            decoration: const BoxDecoration(
              image: DecorationImage(
                image: NetworkImage(
                  'https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=1000&auto=format&fit=crop',
                ),
                fit: BoxFit.cover,
              ),
            ),
          ),
          // Dark Overlay
          Container(color: Colors.black.withOpacity(0.4)),
          // Content
          SafeArea(
            child: Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(24.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 20),

                    // Glassmorphism Card
                    ClipRRect(
                      borderRadius: BorderRadius.circular(20),
                      child: BackdropFilter(
                        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                        child: Container(
                          padding: const EdgeInsets.all(24),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(
                              color: Colors.white.withOpacity(0.2),
                            ),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
<<<<<<< HEAD
                              const Center(
                                child: Text(
                                  'Sign In',
                                  style: TextStyle(
=======
                              Center(
                                child: Text(
                                  'sign_in'.tr(),
                                  style: const TextStyle(
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
                                    color: Colors.white,
                                    fontSize: 28,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                              const SizedBox(height: 30),

                              // Email Field
<<<<<<< HEAD
                              _buildLabel('Email'),
                              _buildTextField(
                                controller: _emailController,
                                hint: 'Email',
=======
                              _buildLabel('email'.tr()),
                              _buildTextField(
                                controller: _emailController,
                                hint: 'email'.tr(),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
                                keyboardType: TextInputType.emailAddress,
                              ),
                              const SizedBox(height: 16),

                              // Password Field
<<<<<<< HEAD
                              _buildLabel('Password'),
                              _buildTextField(
                                controller: _passwordController,
                                hint: 'Password',
=======
                              _buildLabel('password'.tr()),
                              _buildTextField(
                                controller: _passwordController,
                                hint: 'password'.tr(),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
                                obscureText: true,
                              ),
                              const SizedBox(height: 24),

                              // Login Button
                              SizedBox(
                                width: double.infinity,
                                height: 50,
                                child: ElevatedButton(
                                  onPressed: _isLoading ? null : _login,
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: const Color(
                                      0xFF8F9E8B,
                                    ), // Greenish grey
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(8),
                                    ),
                                  ),
                                  child: _isLoading
                                      ? const CircularProgressIndicator(
                                          color: Colors.white,
                                        )
<<<<<<< HEAD
                                      : const Text(
                                          'Sign In',
                                          style: TextStyle(
=======
                                      : Text(
                                          'sign_in'.tr(),
                                          style: const TextStyle(
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
                                            fontSize: 18,
                                            fontWeight: FontWeight.bold,
                                            color: Colors.black87,
                                          ),
                                        ),
                                ),
                              ),
                              const SizedBox(height: 20),

                              // Create Account Link
                              Center(
                                child: GestureDetector(
                                  onTap: () {
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                        builder: (context) =>
                                            const RegisterScreen(),
                                      ),
                                    );
                                  },
                                  child: RichText(
<<<<<<< HEAD
                                    text: const TextSpan(
                                      text: 'New User ? ',
                                      style: TextStyle(color: Colors.white),
                                      children: [
                                        TextSpan(
                                          text: 'Create Account',
                                          style: TextStyle(
=======
                                    text: TextSpan(
                                      text: 'new_user'.tr(),
                                      style: const TextStyle(
                                        color: Colors.white,
                                      ),
                                      children: [
                                        TextSpan(
                                          text: 'create_account'.tr(),
                                          style: const TextStyle(
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
                                            fontWeight: FontWeight.bold,
                                            decoration:
                                                TextDecoration.underline,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ),
<<<<<<< HEAD
=======
                              const SizedBox(height: 24),
                              Center(
                                child: TextButton(
                                  onPressed: () {
                                    Navigator.of(context).pushReplacement(
                                      MaterialPageRoute(
                                        builder: (context) =>
                                            const HomeScreen(),
                                      ),
                                    );
                                  },
                                  child: Text(
                                    'continue_as_guest'.tr(),
                                    style: const TextStyle(
                                      color: Colors.white70,
                                      fontSize: 16,
                                    ),
                                  ),
                                ),
                              ),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
                            ],
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLabel(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.bold,
          fontSize: 14,
        ),
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String hint,
    bool obscureText = false,
    TextInputType? keyboardType,
    Widget? prefix,
  }) {
    return TextField(
      controller: controller,
      obscureText: obscureText,
      keyboardType: keyboardType,
      decoration: InputDecoration(
        hintText: hint,
        filled: true,
        fillColor: Colors.white,
        prefixIcon: prefix != null
            ? Padding(padding: const EdgeInsets.all(12.0), child: prefix)
            : null,
        prefixIconConstraints: const BoxConstraints(minWidth: 0, minHeight: 0),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide.none,
        ),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 16,
          vertical: 14,
        ),
      ),
    );
  }
}
