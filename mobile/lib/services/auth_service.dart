import 'package:dio/dio.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../constants.dart';

class AuthService {
  final Dio _dio = Dio();
  final FlutterSecureStorage _storage = const FlutterSecureStorage();
  static Map<String, dynamic>? _cachedProfile;

  Future<bool> login(String email, String password) async {
    try {
      final response = await _dio.post(
        '${AppConstants.baseUrl}/auth/login',
        data: {'username': email, 'password': password},
        options: Options(
          contentType: Headers
              .formUrlEncodedContentType, // Sets application/x-www-form-urlencoded
        ),
      );

      if (response.statusCode == 200) {
        final token = response.data['access_token'];
        final name =
            response.data['username'] ?? response.data['user']?['name'];

        if (token != null) {
          await _storage.write(key: 'access_token', value: token);
          if (name != null) {
            await _storage.write(key: 'user_name', value: name);
          }
          return true;
        }
      }
      return false;
    } catch (e) {
      print('Login error: $e');
      return false;
    }
  }

  Future<bool> register(String name, String email, String password) async {
    try {
      final response = await _dio.post(
        '${AppConstants.baseUrl}/auth/register',
        data: {'name': name, 'email': email, 'password': password},
      );

      if (response.statusCode == 200 || response.statusCode == 201) {
        await _storage.write(key: 'user_name', value: name);
        return true;
      }
      return false;
    } catch (e) {
      print('Registration error: $e');
      return false;
    }
  }

  Future<void> logout() async {
    await _storage.delete(key: 'access_token');
    await _storage.delete(key: 'user_name');
    _cachedProfile = null;
  }

  Future<String?> getUserName() async {
    return await _storage.read(key: 'user_name');
  }

  Future<Map<String, dynamic>?> getUserProfile({
    bool forceRefresh = false,
  }) async {
    if (_cachedProfile != null && !forceRefresh) {
      return _cachedProfile;
    }

    try {
      final token = await _storage.read(key: 'access_token');
      if (token == null) return null;

      final response = await _dio.get(
        '${AppConstants.baseUrl}/auth/me',
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );

      if (response.statusCode == 200) {
        _cachedProfile = response.data;
        return response.data;
      }
      return null;
    } catch (e) {
      print('Get profile error: $e');
      return _cachedProfile;
    }
  }

  Future<bool> updateUserProfile(Map<String, dynamic> data) async {
    try {
      final token = await _storage.read(key: 'access_token');
      if (token == null) return false;

      final response = await _dio.put(
        '${AppConstants.baseUrl}/auth/me',
        data: data,
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );

      return response.statusCode == 200;
    } catch (e) {
      print('Update profile error: $e');
      return false;
    }
  }

  Future<bool> uploadProfilePhoto(String filePath) async {
    return _uploadPhoto(filePath, 'profile-photo');
  }

  Future<bool> uploadCoverPhoto(String filePath) async {
    return _uploadPhoto(filePath, 'cover-photo');
  }

  Future<bool> _uploadPhoto(String filePath, String endpointSuffix) async {
    try {
      final token = await _storage.read(key: 'access_token');
      if (token == null) return false;

      String fileName = filePath.split('/').last;
      FormData formData = FormData.fromMap({
        "file": await MultipartFile.fromFile(filePath, filename: fileName),
      });

      final response = await _dio.post(
        '${AppConstants.baseUrl}/auth/me/$endpointSuffix',
        data: formData,
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );

      if (response.statusCode == 200) {
        _cachedProfile = null;
        return true;
      }
      return false;
    } catch (e) {
      print('Upload photo error: $e');
      return false;
    }
  }

  Future<bool> deleteAccount() async {
    try {
      final token = await _storage.read(key: 'access_token');
      if (token == null) return false;

      final response = await _dio.delete(
        '${AppConstants.baseUrl}/auth/me',
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );

      if (response.statusCode == 200 || response.statusCode == 204) {
        await logout();
        return true;
      }
      return false;
    } catch (e) {
      print('Delete account error: $e');
      return false;
    }
  }

  Future<String?> changePassword(
    String currentPassword,
    String newPassword,
  ) async {
    try {
      final token = await _storage.read(key: 'access_token');
      if (token == null) return 'Not authenticated';

      final response = await _dio.post(
        '${AppConstants.baseUrl}/auth/change-password',
        data: {
          'current_password': currentPassword,
          'new_password': newPassword,
        },
        options: Options(
          headers: {
            'Authorization': 'Bearer $token',
            'Content-Type': 'application/json',
          },
        ),
      );

      if (response.statusCode == 200) {
        return null;
      }
      return 'Unknown error';
    } catch (e) {
      if (e is DioException) {
        print('Change password error: ${e.response?.data}');
        print('Status code: ${e.response?.statusCode}');
        final data = e.response?.data;
        if (data is Map && data.containsKey('detail')) {
          return data['detail'].toString();
        }
      } else {
        print('Change password error: $e');
      }
      return 'Failed to change password';
    }
  }
}
