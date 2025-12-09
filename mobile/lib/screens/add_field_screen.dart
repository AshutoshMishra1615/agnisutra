import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:geolocator/geolocator.dart';
import 'package:easy_localization/easy_localization.dart';

class AddFieldScreen extends StatefulWidget {
  final String fieldName;
  final LatLng? existingLocation;

  const AddFieldScreen({
    super.key,
    required this.fieldName,
    this.existingLocation,
  });

  @override
  State<AddFieldScreen> createState() => _AddFieldScreenState();
}

class _AddFieldScreenState extends State<AddFieldScreen> {
  LatLng? _currentLocation;
  bool _isLoadingLocation = true;
  final MapController _mapController = MapController();
  late TextEditingController _nameController;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController(text: widget.fieldName);
    if (widget.existingLocation != null) {
      _currentLocation = widget.existingLocation;
      _isLoadingLocation = false;
    } else {
      _getCurrentLocation();
    }
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  Future<void> _getCurrentLocation() async {
    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        setState(() => _isLoadingLocation = false);
        return;
      }

      Position position = await Geolocator.getCurrentPosition();
      if (mounted) {
        setState(() {
          _currentLocation = LatLng(position.latitude, position.longitude);
          _isLoadingLocation = false;
        });
      }
    } catch (e) {
      debugPrint("Error getting location: $e");
      if (mounted) setState(() => _isLoadingLocation = false);
    }
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
          widget.fieldName,
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Field Name',
              style: TextStyle(color: Colors.white70, fontSize: 14),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _nameController,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                filled: true,
                fillColor: const Color(0xFF3E3E3E),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide.none,
                ),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 14,
                ),
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              'Field Location',
              style: TextStyle(color: Colors.white70, fontSize: 14),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.grey[700],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: _isLoadingLocation
                    ? const Center(
                        child: CircularProgressIndicator(
                          color: Color(0xFFC5E1A5),
                        ),
                      )
                    : _currentLocation == null
                    ? const Center(
                        child: Text(
                          "Location not available",
                          style: TextStyle(color: Colors.white),
                        ),
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: FlutterMap(
                          mapController: _mapController,
                          options: MapOptions(
                            initialCenter: _currentLocation!,
                            initialZoom: 15.0,
                            onTap: (tapPosition, point) {
                              setState(() {
                                _currentLocation = point;
                              });
                            },
                          ),
                          children: [
                            TileLayer(
                              urlTemplate:
                                  'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                              userAgentPackageName: 'com.example.mobile',
                            ),
                            MarkerLayer(
                              markers: [
                                Marker(
                                  point: _currentLocation!,
                                  width: 40,
                                  height: 40,
                                  child: const Icon(
                                    Icons.location_on,
                                    color: Colors.red,
                                    size: 40,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
              ),
            ),
            const SizedBox(height: 32),
            Center(
              child: SizedBox(
                width: 200,
                child: ElevatedButton(
                  onPressed: () {
                    if (_nameController.text.trim().isEmpty) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('Please enter a field name'),
                        ),
                      );
                      return;
                    }
                    // Save logic
                    Navigator.pop(context, {
                      'name': _nameController.text.trim(),
                      'location': _currentLocation,
                    });
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFC5E1A5),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                  ),
                  child: Text(
                    'save'.tr(),
                    style: const TextStyle(
                      color: Colors.black,
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
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
}
