import 'package:flutter/material.dart';
<<<<<<< HEAD
// import 'package:easy_localization/easy_localization.dart';
=======
import 'package:easy_localization/easy_localization.dart';
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
import 'package:image_picker/image_picker.dart';
import '../services/auth_service.dart';
import '../constants.dart';
import 'login_screen.dart';
<<<<<<< HEAD
=======
import 'register_screen.dart';
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final AuthService _authService = AuthService();
  final ImagePicker _picker = ImagePicker();
  Map<String, dynamic>? _userProfile;
  bool _isLoading = false; // Default to false to show Guest UI immediately
  bool _isEditing = false;
  int _imageRefreshKey = 0;

  @override
  void initState() {
    super.initState();
    _checkLoginStatus();
  }

  Future<void> _checkLoginStatus() async {
    final token = await _authService.getToken();
    if (token != null) {
      // Try to get cached profile first without loading spinner
      final cachedProfile = await _authService.getUserProfile(
        forceRefresh: false,
      );

      if (mounted) {
        setState(() {
          _userProfile = cachedProfile;
          // Only show loader if we have NO data at all
          _isLoading = _userProfile == null;
        });
      }

      // If we didn't have cached data, or just to be sure, try fetching fresh data
      // But if we already showed cached data, this happens in background
      if (_userProfile == null) {
        _fetchProfile();
      } else {
        // We have data, but let's try to update it silently in the background
        _authService
            .getUserProfile(forceRefresh: true)
            .then((freshProfile) {
              if (mounted && freshProfile != null) {
                setState(() {
                  _userProfile = freshProfile;
                });
              }
            })
            .catchError((e) {
              // Backend unreachable? No problem, user is already seeing cached data.
              debugPrint("Background profile update failed: $e");
            });
      }
    }
  }

  Future<void> _fetchProfile({bool force = false}) async {
    try {
      final profile = await _authService.getUserProfile(forceRefresh: force);
      if (mounted) {
        setState(() {
          _userProfile = profile;
          _isLoading = false;
        });
      }
    } catch (e) {
      // If fetch fails and we are loading, stop loading
      if (mounted && _isLoading) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _pickAndUploadImage(bool isProfilePhoto) async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image == null) return;

      setState(() => _isLoading = true);

      bool success;
      if (isProfilePhoto) {
        success = await _authService.uploadProfilePhoto(image.path);
      } else {
        success = await _authService.uploadCoverPhoto(image.path);
      }

      if (success) {
        await _fetchProfile(force: true);
        if (mounted) {
          setState(() {
            _imageRefreshKey++;
          });
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                isProfilePhoto
<<<<<<< HEAD
                    ? 'Profile photo updated'
                    : 'Cover photo updated',
=======
                    ? 'profile_photo_updated'.tr()
                    : 'cover_photo_updated'.tr(),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
              ),
              backgroundColor: Colors.green,
            ),
          );
        }
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
<<<<<<< HEAD
            const SnackBar(
              content: Text('Failed to update photo'),
=======
            SnackBar(
              content: Text('failed_update_photo'.tr()),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
              backgroundColor: Colors.red,
            ),
          );
        }
        setState(() => _isLoading = false);
      }
    } catch (e) {
      print('Error picking image: $e');
      setState(() => _isLoading = false);
    }
  }

  Future<void> _confirmDeleteAccount() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF1E1E1E),
<<<<<<< HEAD
        title: const Text(
          'Delete Account',
          style: TextStyle(color: Colors.white),
        ),
        content: const Text(
          'Are you sure you want to delete your account? This action cannot be undone.',
          style: TextStyle(color: Colors.white70),
=======
        title: Text(
          'delete_account'.tr(),
          style: const TextStyle(color: Colors.white),
        ),
        content: Text(
          'delete_account_confirmation'.tr(),
          style: const TextStyle(color: Colors.white70),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
<<<<<<< HEAD
            child: const Text('Cancel', style: TextStyle(color: Colors.white)),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text(
              'Delete',
              style: TextStyle(color: Colors.redAccent),
=======
            child: Text(
              'cancel'.tr(),
              style: const TextStyle(color: Colors.white),
            ),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: Text(
              'delete'.tr(),
              style: const TextStyle(color: Colors.redAccent),
>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
            ),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      setState(() => _isLoading = true);
      final success = await _authService.deleteAccount();
      if (success) {
        if (mounted) {
          Navigator.of(context).pushAndRemoveUntil(
            MaterialPageRoute(builder: (context) => const LoginScreen()),
            (route) => false,
          );
        }
      } else {
        if (mounted) {
          setState(() => _isLoading = false);
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Failed to delete account'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    }
  }

  Future<void> _showChangePasswordDialog() async {
    final currentPasswordController = TextEditingController();
    final newPasswordController = TextEditingController();
    final formKey = GlobalKey<FormState>();

    await showDialog(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: const Color(0xFF1E1E1E),
        title: const Text(
          'Change Password',
          style: TextStyle(color: Colors.white),
        ),
        content: Form(
          key: formKey,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextFormField(
                controller: currentPasswordController,
                obscureText: true,
                style: const TextStyle(color: Colors.white),
                decoration: const InputDecoration(
                  labelText: 'Current Password',
                  labelStyle: TextStyle(color: Colors.white70),
                  enabledBorder: UnderlineInputBorder(
                    borderSide: BorderSide(color: Colors.white24),
                  ),
                ),
                validator: (value) =>
                    value?.isEmpty ?? true ? 'Required' : null,
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: newPasswordController,
                obscureText: true,
                style: const TextStyle(color: Colors.white),
                decoration: const InputDecoration(
                  labelText: 'New Password',
                  labelStyle: TextStyle(color: Colors.white70),
                  enabledBorder: UnderlineInputBorder(
                    borderSide: BorderSide(color: Colors.white24),
                  ),
                ),
                validator: (value) =>
                    (value?.length ?? 0) < 6 ? 'Min 6 chars' : null,
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Cancel', style: TextStyle(color: Colors.white)),
          ),
          TextButton(
            onPressed: () async {
              if (formKey.currentState?.validate() ?? false) {
                Navigator.pop(dialogContext); // Close dialog first
                setState(() => _isLoading = true);

                final errorMessage = await _authService.changePassword(
                  currentPasswordController.text,
                  newPasswordController.text,
                );

                if (mounted) {
                  setState(() => _isLoading = false);
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text(
                        errorMessage == null
                            ? 'Password changed successfully'
                            : errorMessage,
                      ),
                      backgroundColor: errorMessage == null
                          ? Colors.green
                          : Colors.red,
                    ),
                  );
                }
              }
            },
            child: const Text(
              'Change',
              style: TextStyle(color: Color(0xFF8F9E8B)),
            ),
          ),
        ],
      ),
    );
  }

<<<<<<< HEAD
=======
  Widget _buildGuestProfile() {
    return Scaffold(
      backgroundColor: const Color(0xFF1E1E1E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E1E1E),
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Profile',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      body: Column(
        children: [
          SizedBox(
            height: 240,
            child: Stack(
              alignment: Alignment.center,
              children: [
                Positioned(
                  top: 0,
                  left: 0,
                  right: 0,
                  height: 160,
                  child: Container(color: const Color(0xFF616161)),
                ),
                Positioned(
                  top: 110,
                  child: Container(
                    padding: const EdgeInsets.all(4),
                    decoration: const BoxDecoration(
                      color: Color(0xFF1E1E1E),
                      shape: BoxShape.circle,
                    ),
                    child: const CircleAvatar(
                      radius: 50,
                      backgroundColor: Colors.grey,
                      child: Icon(Icons.person, size: 60, color: Colors.white),
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),
          const Text(
            'Guest Account',
            style: TextStyle(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Joined on ${DateFormat('dd MMM yyyy').format(DateTime.now())}',
            style: const TextStyle(color: Colors.grey, fontSize: 14),
          ),
          const SizedBox(height: 40),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 40),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const LoginScreen(),
                        ),
                      );
                    },
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: Colors.white),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    child: const Text(
                      'Sign In',
                      style: TextStyle(color: Colors.white),
                    ),
                  ),
                ),
                const SizedBox(width: 20),
                Expanded(
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const RegisterScreen(),
                        ),
                      );
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFC5E1A5),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    child: const Text(
                      'Sign Up',
                      style: TextStyle(color: Colors.black),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        backgroundColor: Color(0xFF050F06),
        body: Center(
          child: CircularProgressIndicator(color: Color(0xFF8F9E8B)),
        ),
      );
    }

<<<<<<< HEAD
=======
    if (_userProfile == null) {
      return _buildGuestProfile();
    }

>>>>>>> eb9d84b43aa988147346dc664959429ed6a207b3
    final name = _userProfile?['name'] ?? 'User';
    final role = _userProfile?['role'] ?? 'Farmer';
    final email = _userProfile?['email'] ?? 'No email';
    // final city = _userProfile?['city'] ?? 'Unknown Location';
    // final deviceId = _userProfile?['device_id'] ?? 'Not Linked';
    final joinedDate = _userProfile?['created_at'] != null
        ? DateTime.parse(
            _userProfile!['created_at'],
          ).toString().substring(0, 10)
        : 'Unknown';

    String? profilePhoto = _userProfile?['profile_photo'];
    String? coverPhoto = _userProfile?['cover_photo'];

    // Helper to construct full URL
    String? getFullUrl(String? path) {
      if (path == null) return null;
      if (path.startsWith('http')) return '$path?v=$_imageRefreshKey';
      return '${AppConstants.baseUrl}$path?v=$_imageRefreshKey';
    }

    profilePhoto = getFullUrl(profilePhoto);
    coverPhoto = getFullUrl(coverPhoto);

    return Scaffold(
      backgroundColor: const Color(0xFF050F06), // Very dark green background
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Profile',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Header Section with Stack
            SizedBox(
              height: 220,
              child: Stack(
                children: [
                  // Cover Image
                  Positioned(
                    top: 0,
                    left: 0,
                    right: 0,
                    height: 160,
                    child: GestureDetector(
                      onTap: _isEditing
                          ? () => _pickAndUploadImage(false)
                          : null,
                      child: Container(
                        decoration: BoxDecoration(
                          image: DecorationImage(
                            image: coverPhoto != null
                                ? NetworkImage(coverPhoto)
                                : const NetworkImage(
                                    'https://images.unsplash.com/photo-1500382017468-9049fed747ef?q=80&w=1000&auto=format&fit=crop',
                                  ), // Farm cover fallback
                            fit: BoxFit.cover,
                          ),
                        ),
                        child: _isEditing
                            ? Align(
                                alignment: Alignment.topRight,
                                child: Padding(
                                  padding: const EdgeInsets.all(8.0),
                                  child: Container(
                                    padding: const EdgeInsets.all(4),
                                    decoration: BoxDecoration(
                                      color: Colors.black.withOpacity(0.5),
                                      shape: BoxShape.circle,
                                    ),
                                    child: const Icon(
                                      Icons.edit,
                                      color: Colors.white,
                                      size: 20,
                                    ),
                                  ),
                                ),
                              )
                            : null,
                      ),
                    ),
                  ),
                  // Gradient Overlay
                  Positioned(
                    top: 0,
                    left: 0,
                    right: 0,
                    height: 160,
                    child: IgnorePointer(
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                            colors: [
                              Colors.transparent,
                              Colors.black.withOpacity(0.3),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                  // Profile Image
                  Positioned(
                    bottom: 0,
                    left: 20,
                    child: GestureDetector(
                      onTap: _isEditing
                          ? () => _pickAndUploadImage(true)
                          : null,
                      child: Stack(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(4),
                            decoration: const BoxDecoration(
                              color: Color(0xFF050F06), // Match background
                              shape: BoxShape.circle,
                            ),
                            child: CircleAvatar(
                              radius: 50,
                              backgroundImage: profilePhoto != null
                                  ? NetworkImage(profilePhoto)
                                  : const NetworkImage(
                                      'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=200&auto=format&fit=crop',
                                    ), // Farmer face fallback
                            ),
                          ),
                          if (_isEditing)
                            Positioned(
                              bottom: 0,
                              right: 0,
                              child: Container(
                                padding: const EdgeInsets.all(6),
                                decoration: const BoxDecoration(
                                  color: Color(0xFF8F9E8B),
                                  shape: BoxShape.circle,
                                ),
                                child: const Icon(
                                  Icons.camera_alt,
                                  size: 16,
                                  color: Colors.black,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
            // User Info
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    name,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    role.toString().toUpperCase(),
                    style: const TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                  Text(
                    'Joined on $joinedDate',
                    style: const TextStyle(color: Colors.white54, fontSize: 12),
                  ),
                  const SizedBox(height: 20),

                  // Action Buttons
                  Row(
                    children: [
                      OutlinedButton(
                        onPressed: _showChangePasswordDialog,
                        style: OutlinedButton.styleFrom(
                          side: const BorderSide(color: Color(0xFF8F9E8B)),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                          padding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 8,
                          ),
                        ),
                        child: const Text(
                          'Change Password',
                          style: TextStyle(color: Color(0xFF8F9E8B)),
                        ),
                      ),
                      const SizedBox(width: 12),
                      ElevatedButton(
                        onPressed: () {
                          setState(() {
                            _isEditing = !_isEditing;
                          });
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF8F9E8B),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                          padding: const EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 8,
                          ),
                        ),
                        child: Text(
                          _isEditing ? 'Done' : 'Edit Profile',
                          style: const TextStyle(color: Colors.black),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 40),

                  // Details List
                  _buildDetailRow('NAME', name),
                  const Divider(color: Colors.white24),
                  _buildDetailRow('EMAIL ID', email),
                  const Divider(color: Colors.white24),
                  // _buildDetailRow('LOCATION', city),
                  // const Divider(color: Colors.white24),

                  // _buildDetailRow('DEVICE ID', deviceId),
                  const SizedBox(height: 60),

                  // Footer Buttons
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton(
                          onPressed: () async {
                            await _authService.logout();
                            if (context.mounted) {
                              Navigator.of(context).pushAndRemoveUntil(
                                MaterialPageRoute(
                                  builder: (context) => const LoginScreen(),
                                ),
                                (route) => false,
                              );
                            }
                          },
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: Colors.redAccent),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                          child: const Text(
                            'Logout',
                            style: TextStyle(
                              color: Colors.redAccent,
                              fontSize: 16,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: ElevatedButton(
                          onPressed: _confirmDeleteAccount,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.redAccent,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                          child: const Text(
                            'Delete Account',
                            style: TextStyle(color: Colors.white, fontSize: 16),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 40),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
          Text(
            value,
            style: const TextStyle(color: Colors.white, fontSize: 14),
          ),
        ],
      ),
    );
  }
}
