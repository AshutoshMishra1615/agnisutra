import 'package:dio/dio.dart';
import '../constants.dart';

class NdviService {
  // Replace with your backend URL (use 10.0.2.2 for Android emulator to access localhost)
  static const String baseUrl = AppConstants.baseUrl + '/krishi-saathi';

  final Dio _dio = Dio(
    BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 10),
      receiveTimeout: const Duration(seconds: 10),
    ),
  );

  Future<Map<String, dynamic>> fetchNdvi(double lat, double lon) async {
    try {
      final response = await _dio.get(
        '/ndvi',
        queryParameters: {'lat': lat, 'lon': lon},
      );

      if (response.statusCode == 200) {
        return response.data as Map<String, dynamic>;
      } else {
        throw Exception('Failed to load NDVI data: ${response.statusCode}');
      }
    } on DioException catch (e) {
      // Handle Dio-specific errors (timeout, connection refused, etc.)
      if (e.response != null) {
        throw Exception('Server error: ${e.response?.data ?? e.message}');
      } else {
        throw Exception('Network error: ${e.message}');
      }
    } catch (e) {
      throw Exception('Error fetching NDVI: $e');
    }
  }
}
