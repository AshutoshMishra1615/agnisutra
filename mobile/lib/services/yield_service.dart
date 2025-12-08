import 'package:dio/dio.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:hive_flutter/hive_flutter.dart';
import '../constants.dart';

class YieldService {
  final Dio _dio = Dio();
  final FlutterSecureStorage _storage = const FlutterSecureStorage();

  Future<Map<String, dynamic>?> getPrediction(Map<String, dynamic> formData) async {
    var box = await Hive.openBox('last_prediction');
    
    try {
      String? token = await _storage.read(key: 'access_token');
      
      final response = await _dio.post(
        '${AppConstants.baseUrl}/predict', 
        data: formData,
        options: Options(
          headers: {
            'Authorization': 'Bearer $token',
          },
        ),
      );

      if (response.statusCode == 200) {
        // Save to Hive
        await box.put('data', response.data);
        await box.put('is_offline', false);
        return response.data;
      }
      return null;
    } catch (e) {
      print('Prediction error: $e');
      // Load from Hive if available
      if (box.containsKey('data')) {
        final data = Map<String, dynamic>.from(box.get('data'));
        data['is_offline'] = true; // Flag to show UI
        return data;
      }
      return null;
    }
  }
}
