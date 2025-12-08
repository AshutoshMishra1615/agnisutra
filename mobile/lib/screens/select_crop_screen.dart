import 'package:flutter/material.dart';

class SelectCropScreen extends StatelessWidget {
  const SelectCropScreen({super.key});

  final List<Map<String, String>> crops = const [
    {'name': 'Sunflower', 'icon': 'assets/images/icons/Frame 264.png'},
    {'name': 'Soybean', 'icon': 'assets/images/icons/Frame 265.png'},
    {'name': 'Mustard', 'icon': 'assets/images/icons/Frame 266.png'},
    {'name': 'Groundnut', 'icon': 'assets/images/icons/Frame 267.png'},
    {'name': 'Sesame', 'icon': 'assets/images/icons/Frame 268.png'},
    {'name': 'Castor', 'icon': 'assets/images/icons/Frame 264 (1).png'},
    {'name': 'Safflower', 'icon': 'assets/images/icons/Frame 267 (1).png'},
    {'name': 'Niger', 'icon': 'assets/images/icons/Frame 267 (2).png'},
  ];

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
        title: const Text(
          'Select Crops',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Add the crops that you want to add.',
              style: TextStyle(color: Colors.white70, fontSize: 16),
            ),
            const SizedBox(height: 24),
            Expanded(
              child: GridView.builder(
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 3,
                  crossAxisSpacing: 16,
                  mainAxisSpacing: 24,
                  childAspectRatio: 0.8,
                ),
                itemCount: crops.length,
                itemBuilder: (context, index) {
                  final crop = crops[index];
                  return GestureDetector(
                    onTap: () {
                      Navigator.pop(context, crop['name']);
                    },
                    child: Column(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: const Color(0xFFEBC25C),
                              width: 2,
                            ),
                            color: const Color(0xFF1E1E1E),
                          ),
                          child: Image.asset(
                            crop['icon']!,
                            height: 40,
                            width: 40,
                            errorBuilder: (context, error, stackTrace) =>
                                const Icon(
                                  Icons.grass,
                                  color: Colors.white,
                                  size: 40,
                                ),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          crop['name']!,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
