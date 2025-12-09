import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:geolocator/geolocator.dart';
import 'package:geocoding/geocoding.dart';
import '../services/yield_service.dart';
// import '../models/crop_data.dart';
import 'recommended_fertilizer_screen.dart';

class AddCropDetailsScreen extends StatefulWidget {
  final String cropName;

  const AddCropDetailsScreen({super.key, required this.cropName});

  @override
  State<AddCropDetailsScreen> createState() => _AddCropDetailsScreenState();
}

class _AddCropDetailsScreenState extends State<AddCropDetailsScreen> {
  // final _maturityController = TextEditingController();
  final _districtController = TextEditingController();
  final _pinController = TextEditingController();
  final _stateController = TextEditingController();

  // New controllers for Fertilizer Recommendation
  final _targetYieldController = TextEditingController();
  final _soilNController = TextEditingController();
  final _soilPController = TextEditingController();
  final _soilKController = TextEditingController();
  final _tempController = TextEditingController();
  final _moistureController = TextEditingController();

  final _yieldService = YieldService();
  bool _isLoadingLocation = false;
  bool _isPredicting = false;
  // bool _showAdvanced = false;

  @override
  void initState() {
    super.initState();
    _getCurrentLocationAndData();
  }

  Future<void> _getCurrentLocationAndData() async {
    setState(() {
      _isLoadingLocation = true;
    });

    try {
      // 1. Get Location
      bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) throw 'Location services are disabled.';

      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
        if (permission == LocationPermission.denied)
          throw 'Location permissions are denied';
      }
      if (permission == LocationPermission.deniedForever)
        throw 'Location permissions are permanently denied.';

      Position position = await Geolocator.getCurrentPosition();

      // Fill Address
      try {
        List<Placemark> placemarks = await placemarkFromCoordinates(
          position.latitude,
          position.longitude,
        );
        if (placemarks.isNotEmpty) {
          Placemark place = placemarks[0];
          if (mounted) {
            setState(() {
              _districtController.text =
                  place.subAdministrativeArea ?? place.locality ?? '';
              _pinController.text = place.postalCode ?? '';
              _stateController.text = place.administrativeArea ?? '';
            });
          }
        }
      } catch (e) {
        debugPrint('Error getting address: $e');
      }

      // 2. Fetch Weather Data
      try {
        final weatherData = await _yieldService.getWeatherData(
          position.latitude,
          position.longitude,
        );
        if (weatherData != null && mounted) {
          setState(() {
            if (weatherData['stats'] != null &&
                weatherData['stats']['mean_temp_gs_C'] != null) {
              _tempController.text = weatherData['stats']['mean_temp_gs_C']
                  .toString();
            } else {
              _tempController.text = (weatherData['temperature'] ?? '')
                  .toString();
            }
            // Fallback moisture from humidity if IoT fails
            if (_moistureController.text.isEmpty) {
              _moistureController.text = (weatherData['humidity'] ?? '')
                  .toString();
            }
          });
        }
      } catch (e) {
        debugPrint('Error fetching weather: $e');
      }

      // 3. Fetch IoT Data (Soil)
      try {
        final iotData = await _yieldService.getIoTData();
        if (iotData != null && mounted) {
          setState(() {
            _soilNController.text = (iotData['nitrogen'] ?? '').toString();
            _soilPController.text = (iotData['phosphorus'] ?? '').toString();
            _soilKController.text = (iotData['potassium'] ?? '').toString();
            _moistureController.text = (iotData['moisture'] ?? '').toString();
          });
        }
      } catch (e) {
        debugPrint('Error fetching IoT data: $e');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoadingLocation = false;
        });
      }
    }
  }

  @override
  void dispose() {
    // _maturityController.dispose();
    _districtController.dispose();
    _pinController.dispose();
    _stateController.dispose();
    _targetYieldController.dispose();
    _soilNController.dispose();
    _soilPController.dispose();
    _soilKController.dispose();
    _tempController.dispose();
    _moistureController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1E1E1E),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          'add_crop'.tr(),
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // _buildSectionTitle('crop_details'.tr()),
            // const SizedBox(height: 16),
            // _buildLabel('maturity_days'.tr()),
            // _buildTextField(_maturityController),
            const SizedBox(height: 24),

            _buildSectionTitle('location'.tr()),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildLabel('district'.tr()),
                      _buildTextField(_districtController),
                    ],
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildLabel('pin'.tr()),
                      _buildTextField(_pinController),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _buildLabel('state'.tr()),
            _buildTextField(_stateController),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _isLoadingLocation ? null : _getCurrentLocationAndData,
              icon: _isLoadingLocation
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.black,
                      ),
                    )
                  : const Icon(Icons.refresh, color: Colors.black),
              label: Text(
                _isLoadingLocation ? 'syncing_data'.tr() : 'refresh_data'.tr(),
                style: const TextStyle(color: Colors.black),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFC5E1A5),
                disabledBackgroundColor: const Color(0xFFC5E1A5),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
              ),
            ),
            const SizedBox(height: 24),

            _buildSectionTitle('yield_goal'.tr()),
            const SizedBox(height: 16),
            _buildLabel('target_yield'.tr()),
            _buildTextField(_targetYieldController, hint: '2.5'),
            const SizedBox(height: 24),

            const SizedBox(height: 32),
            Center(
              child: ElevatedButton.icon(
                onPressed: _isPredicting ? null : _getRecommendation,
                icon: _isPredicting
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.black,
                        ),
                      )
                    : const Icon(Icons.spa, color: Colors.black),
                label: Text(
                  _isPredicting
                      ? 'Calculating...'
                      : 'Get Fertilizer Recommendation',
                  style: const TextStyle(
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFC5E1A5),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 32,
                    vertical: 16,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Future<void> _getRecommendation() async {
    if (_targetYieldController.text.isEmpty ||
        _soilNController.text.isEmpty ||
        _soilPController.text.isEmpty ||
        _soilKController.text.isEmpty ||
        _tempController.text.isEmpty ||
        _moistureController.text.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Please fill all fields')));
      return;
    }

    setState(() => _isPredicting = true);
    try {
      final result = await _yieldService.getFertilizerRecommendation(
        crop: widget.cropName,
        targetYield: double.parse(_targetYieldController.text),
        soilN: double.parse(_soilNController.text),
        soilP: double.parse(_soilPController.text),
        soilK: double.parse(_soilKController.text),
        temperature: double.parse(_tempController.text),
        moisture: double.parse(_moistureController.text),
      );

      if (result == null) throw "Recommendation failed";

      if (!mounted) return;

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => RecommendedFertilizerScreen(
            result: result,
            cropName: widget.cropName,
            // maturityDays: _maturityController.text,
          ),
        ),
      ).then((result) {
        if (result != null && mounted) {
          Navigator.pop(context, result);
        }
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    } finally {
      if (mounted) setState(() => _isPredicting = false);
    }
  }

  // Widget _buildResultItem(String label, dynamic value) {
  //   return Column(
  //     children: [
  //       Text(
  //         value.toString(),
  //         style: const TextStyle(
  //           color: Color(0xFFC5E1A5),
  //           fontSize: 24,
  //           fontWeight: FontWeight.bold,
  //         ),
  //       ),
  //       const SizedBox(height: 4),
  //       Text(
  //         label,
  //         style: const TextStyle(color: Colors.white70, fontSize: 12),
  //       ),
  //     ],
  //   );
  // }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(
        color: Colors.white,
        fontSize: 16,
        fontWeight: FontWeight.bold,
      ),
    );
  }

  Widget _buildLabel(String label) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Text(
        label,
        style: const TextStyle(color: Colors.white70, fontSize: 14),
      ),
    );
  }

  Widget _buildTextField(TextEditingController controller, {String? hint}) {
    return TextField(
      controller: controller,
      style: const TextStyle(color: Colors.white),
      keyboardType: TextInputType.number,
      decoration: InputDecoration(
        filled: true,
        fillColor: const Color(0xFF3E3E3E),
        hintText: hint,
        hintStyle: const TextStyle(color: Colors.white30),
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
