/*
 * Stratified Entropy Permutation Tool.
 * Copyright (c) 2026 AlleSerOjje.
*/


#include <sstream>
#include <cstring>
#include "Timer.h"
#include "Vanity.h"
#include "SECP256k1.h"
#include "GPU/GPUEngine.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>
#include "hash/sha512.h"
#include "hash/sha256.h"
#include <thread>
#include <atomic>
#include <iostream>
#include <cmath>

#if defined(_WIN32) || defined(_WIN64)
#include <conio.h> // For _kbhit and _getch on Windows
#else
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif

std::atomic<bool> Pause(false);
std::atomic<bool> Paused(false);
std::atomic<bool> stopMonitorKey(false);
int idxcount;
double t_Paused;
bool randomMode = false;

#if defined(_WIN32) || defined(_WIN64)
void monitorKeypress() {
	while (!stopMonitorKey) {
		Timer::SleepMillis(1);
		if (_kbhit()) { // Check if a key is pressed
			char ch = _getch(); // Get the pressed key
			if (ch == 'p' || ch == 'P') {
				Pause = !(Pause);
				//printf("\nPause PRESSED\n");
				//break;
			}
		}
	}
}
#else
void setTerminalRawMode(bool enable) {
	static struct termios oldt, newt;
	if (enable) {
		tcgetattr(STDIN_FILENO, &oldt); // Get current terminal settings
		newt = oldt;
		newt.c_lflag &= ~(ICANON | ECHO); // Disable canonical mode and echo
		tcsetattr(STDIN_FILENO, TCSANOW, &newt); // Apply the new settings
	}
	else {
		tcsetattr(STDIN_FILENO, TCSANOW, &oldt); // Restore old settings
	}
}

void setNonBlockingInput(bool enable) {
	int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
	if (enable) {
		fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK); // Modalit� non bloccante
	}
	else {
		fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK); // Ripristina modalit� bloccante
	}
}

void monitorKeypress() {
	setTerminalRawMode(true);
	setNonBlockingInput(true);  // Imposta stdin in modalit� non bloccante

	while (!stopMonitorKey) {
		Timer::SleepMillis(1);
		char ch;
		if (read(STDIN_FILENO, &ch, 1) > 0) { // Ora non blocca
			if (ch == 'p' || ch == 'P') {
				Pause = !Pause;
			}
		}
	}

	setNonBlockingInput(false);  // Ripristina modalit� normale
	setTerminalRawMode(false);
}
#endif


#define RELEASE "2.2"

using namespace std;

// ------------------------------------------------------------------------------------------

void printUsage() {

	printf("StringCrack [-v] [-gpuId] [-i inputfile] [-o outputfile] [-start HEX] [-range] [-end] [-m] [-stop] [-random]\n");
	printf("            [-lock \"pos:val,...\"] [-popcount N] [-poprange min:max]\n \n");
	printf(" -v: Print version\n");
	printf(" -i inputfile: Get list of addresses to search from specified file\n");
	printf(" -o outputfile: Output results to the specified file\n");
	printf(" -gpuId: GPU to use, default is 0\n");
	printf(" -start start Private Key HEX\n");
	printf(" -range bit range dimension. start -> (start + 2^range).\n");
	printf(" -end bit sub-range limit. scan 2^end seeds from start (for parallelization).\n");
	printf(" -m: Max number of prefixes found by each kernel call, default is 262144 (use multiple of 65536)\n");
	printf(" -stop: Stop when all prefixes are found\n");
	printf(" -random: Random mode active.\n");
	printf(" -backup: Backup mode.\n");
	printf("\n === StringCrack Mode ===\n");
	printf(" -lock \"pos:val,...\": Lock bit positions. Example: -lock \"93:0,98:0,99:0,78:0\"\n");
	printf(" -popcount N: Target popcount. Example: -popcount 37\n");
	printf(" -poprange min:max: Popcount range. Example: -poprange 30:40\n");
	printf(" -center STRING: SEP center string (71-bit). Example: -center 10110...\n");
	printf(" -seprange min:max: SEP mutation range. Example: -seprange 11:23\n");
	printf("\n === SEP6 Radius Mode (GPU-native Gosper — combinatorial unranking) ===\n");
	printf(" -radius N: Test only seeds up to N Hamming-distance from center.\n");
	printf("            GPU computes combinations in registers — zero CPU bottleneck.\n");
	printf("            Popcount pre-filter via -poprange runs BEFORE EC math.\n");
	printf("            Example: -center 10011...10110 -radius 20 -poprange 35:39\n");
	printf(" -radiusrange min:max: Test a specific Hamming distance band.\n");
	printf("            Jumps straight to layer 'min', skipping layers 0..min-1.\n");
	printf("            Example: -center 10011...10110 -radiusrange 13:21\n");
	printf("\n === SEPHOLY: Shekinah Matrix Mode (Dyadic Decomposition + MITM) ===\n");
	printf(" -shekinah: Enable the Shekinah Matrix decomposition pipeline.\n");
	printf("            Decomposes arbitrary key corridors into optimal power-of-2 blocks,\n");
	printf("            each fed to the MITM God Engine for O(sqrt(N)) coverage.\n");
	printf(" -corridorstart HEX: Start of the multiplier corridor (inclusive).\n");
	printf(" -corridorend HEX: End of the multiplier corridor (exclusive).\n");
	printf("            Example: -shekinah -corridorstart 1A2B3C -corridorend 1A2C00 -poprange 30:40\n");
	exit(-1);

}

// Parse -lock argument: "93:0,98:0,99:0,78:0"
void parseLockString(const string& lockStr, StringCrackConfig* config) {
	config->numLockedBits = 0;
	if (lockStr.empty()) return;
	stringstream ss(lockStr);
	string token;
	while (getline(ss, token, ',')) {
		size_t start = token.find_first_not_of(" \t");
		size_t end = token.find_last_not_of(" \t");
		if (start == string::npos) continue;
		token = token.substr(start, end - start + 1);
		size_t colonPos = token.find(':');
		if (colonPos == string::npos) { fprintf(stderr, "[ERROR] Invalid lock: '%s'\n", token.c_str()); exit(-1); }
		int pos = stoi(token.substr(0, colonPos));
		int val = stoi(token.substr(colonPos + 1));
		if (pos < 0 || pos > 255) { fprintf(stderr, "[ERROR] Lock pos %d out of range\n", pos); exit(-1); }
		if (val != 0 && val != 1) { fprintf(stderr, "[ERROR] Lock val must be 0 or 1\n"); exit(-1); }
		if (config->numLockedBits >= MAX_LOCKED_BITS) { fprintf(stderr, "[ERROR] Too many locked bits\n"); exit(-1); }
		config->lockedBits[config->numLockedBits].position = pos;
		config->lockedBits[config->numLockedBits].value = val;
		config->numLockedBits++;
	}
	printf("[StringCrack] Parsed %d locked bits\n", config->numLockedBits);
	for (int i = 0; i < config->numLockedBits; i++)
		printf("  Bit %d = %d\n", config->lockedBits[i].position, config->lockedBits[i].value);
	fflush(stdout);
}


int getInt(string name, char* v) {

	int r;

	try {

		r = std::stoi(string(v));

	}
	catch (std::invalid_argument&) {

		fprintf(stderr, "[ERROR] Invalid %s argument, number expected\n", name.c_str());
		exit(-1);
	}

	return r;
}

void getInts(string name, vector<int>& tokens, const string& text, char sep) {

	size_t start = 0, end = 0;
	tokens.clear();
	int item;

	try {

		while ((end = text.find(sep, start)) != string::npos) {
			item = std::stoi(text.substr(start, end - start));
			tokens.push_back(item);
			start = end + 1;
		}

		item = std::stoi(text.substr(start));
		tokens.push_back(item);

	}
	catch (std::invalid_argument&) {

		fprintf(stderr, "[ERROR] Invalid %s argument, number expected\n", name.c_str());
		exit(-1);
	}
}

void getKeySpace(const string& text, BITCRACK_PARAM* bc, Int& maxKey)
{
	size_t start = 0, end = 0;
	string item;

	try
	{
		if ((end = text.find(':', start)) != string::npos)
		{
			item = std::string(text.substr(start, end));
			start = end + 1;
		}
		else
		{
			item = std::string(text);
		}

		if (item.length() == 0)
		{
			bc->ksStart.SetInt32(1);
		}
		else if (item.length() > 64)
		{
			fprintf(stderr, "[ERROR] keyspaceSTART: invalid privkey (64 length)\n");
			exit(-1);
		}
		else
		{
			item.insert(0, 64 - item.length(), '0');
			for (int i = 0; i < 32; i++)
			{
				unsigned char my1ch = 0;
				if (sscanf(&item[2 * i], "%02hhX", &my1ch)) {};
				bc->ksStart.SetByte(31 - i, my1ch);
			}
		}

		if (start != 0 && (end = text.find('+', start)) != string::npos)
		{
			item = std::string(text.substr(end + 1));
			if (item.length() > 64 || item.length() == 0)
			{
				fprintf(stderr, "[ERROR] keyspace__END: invalid privkey (64 length)\n");
				exit(-1);
			}

			item.insert(0, 64 - item.length(), '0');

			for (int i = 0; i < 32; i++)
			{
				unsigned char my1ch = 0;
				if (sscanf(&item[2 * i], "%02hhX", &my1ch)) {};
				bc->ksFinish.SetByte(31 - i, my1ch);
			}

			bc->ksFinish.Add(&bc->ksStart);
		}
		else if (start != 0)
		{
			item = std::string(text.substr(start));

			if (item.length() > 64 || item.length() == 0)
			{
				fprintf(stderr, "[ERROR] keyspace__END: invalid privkey (64 length)\n");
				exit(-1);
			}

			item.insert(0, 64 - item.length(), '0');

			for (int i = 0; i < 32; i++)
			{
				unsigned char my1ch = 0;
				if (scanf(&item[2 * i], "%02hhX", &my1ch)) {};
				bc->ksFinish.SetByte(31 - i, my1ch);
			}
		}
		else
		{
			bc->ksFinish.Set(&maxKey);
		}
	}
	catch (std::invalid_argument&)
	{
		fprintf(stderr, "[ERROR] Invalid --keyspace argument \n");
		exit(-1);
	}
}

void checkKeySpace(BITCRACK_PARAM* bc, Int& maxKey)
{
	if (bc->ksStart.IsGreater(&maxKey) || bc->ksFinish.IsGreater(&maxKey))
	{
		fprintf(stderr, "[ERROR] START/END IsGreater %s \n", maxKey.GetBase16().c_str());
		exit(-1);
	}

	if (bc->ksFinish.IsLowerOrEqual(&bc->ksStart))
	{
		fprintf(stderr, "[ERROR] END IsLowerOrEqual START \n");
		exit(-1);
	}

	if (bc->ksFinish.IsLowerOrEqual(&bc->ksNext))
	{
		fprintf(stderr, "[ERROR] END: IsLowerOrEqual NEXT \n");
		exit(-1);
	}

	return;
}

void parseFile(string fileName, vector<string>& lines) {

	// Get file size
	FILE* fp = fopen(fileName.c_str(), "rb");
	if (fp == NULL) {
		fprintf(stderr, "[ERROR] ParseFile: cannot open %s %s\n", fileName.c_str(), strerror(errno));
		exit(-1);
	}
	fseek(fp, 0L, SEEK_END);
	size_t sz = ftell(fp);
	size_t nbAddr = sz / 33; /* Upper approximation */
	bool loaddingProgress = sz > 100000;
	fclose(fp);

	// Parse file
	int nbLine = 0;
	string line;
	ifstream inFile(fileName);
	lines.reserve(nbAddr);
	while (getline(inFile, line)) {

		// Remove ending \r\n
		int l = (int)line.length() - 1;
		while (l >= 0 && isspace(line.at(l))) {
			line.pop_back();
			l--;
		}

		if (line.length() > 0) {
			lines.push_back(line);
			nbLine++;
			if (loaddingProgress) {
				if ((nbLine % 50000) == 0)
					fprintf(stdout, "[Loading input file %5.1f%%]\r", ((double)nbLine * 100.0) / ((double)(nbAddr) * 33.0 / 34.0));
			}
		}
	}

	if (loaddingProgress)
		fprintf(stdout, "[Loading input file 100.0%%]\n");
}

void generateKeyPair(Secp256K1* secp, string seed, int searchMode, bool paranoiacSeed) {

	if (seed.length() < 8) {
		fprintf(stderr, "[ERROR] Use a seed of at leats 8 characters to generate a key pair\n");
		fprintf(stderr, "Ex: VanitySearch -s \"A Strong Password\" -kp\n");
		exit(-1);
	}

	if (searchMode == SEARCH_BOTH) {
		fprintf(stderr, "[ERROR] Use compressed or uncompressed to generate a key pair\n");
		exit(-1);
	}

	bool compressed = (searchMode == SEARCH_COMPRESSED);

	string salt = "0";
	unsigned char hseed[64];
	pbkdf2_hmac_sha512(hseed, 64, (const uint8_t*)seed.c_str(), seed.length(),
		(const uint8_t*)salt.c_str(), salt.length(),
		2048);

	Int privKey;
	privKey.SetInt32(0);
	sha256(hseed, 64, (unsigned char*)privKey.bits64);
	Point p = secp->ComputePublicKey(&privKey);
	fprintf(stdout, "Priv : %s\n", secp->GetPrivAddress(compressed, privKey).c_str());
	fprintf(stdout, "Pub  : %s\n", secp->GetPublicKeyHex(compressed, p).c_str());
}

void outputAdd(string outputFile, int addrType, string addr, string pAddr, string pAddrHex) {

	FILE* f = stdout;
	bool needToClose = false;

	if (outputFile.length() > 0) {
		f = fopen(outputFile.c_str(), "a");
		if (f == NULL) {
			fprintf(stderr, "Cannot open %s for writing\n", outputFile.c_str());
			f = stdout;
		}
		else {
			needToClose = true;
		}
	}

	fprintf(f, "\nPublic Addr: %s\n", addr.c_str());

	switch (addrType) {
	case P2PKH:
		fprintf(f, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
		break;
	case P2SH:
		fprintf(f, "Priv (WIF): p2wpkh-p2sh:%s\n", pAddr.c_str());
		break;
	case BECH32:
		fprintf(f, "Priv (WIF): p2wpkh:%s\n", pAddr.c_str());
		break;
	}
	fprintf(f, "Priv (HEX): 0x%s\n", pAddrHex.c_str());

	if (needToClose)
		fclose(f);
}

#define CHECK_ADDR()                                           \
  fullPriv.ModAddK1order(&e, &partialPrivKey);                 \
  p = secp->ComputePublicKey(&fullPriv);                       \
  cAddr = secp->GetAddress(addrType, compressed, p);           \
  if (cAddr == addr) {                                         \
    found = true;                                              \
    string pAddr = secp->GetPrivAddress(compressed, fullPriv); \
    string pAddrHex = fullPriv.GetBase16();                    \
    outputAdd(outputFile, addrType, addr, pAddr, pAddrHex);    \
  }

void reconstructAdd(Secp256K1* secp, string fileName, string outputFile, string privAddr) {

	bool compressed;
	int addrType;
	Int lambda;
	Int lambda2;
	lambda.SetBase16("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
	lambda2.SetBase16("ac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce");

	Int privKey = secp->DecodePrivateKey((char*)privAddr.c_str(), &compressed);
	if (privKey.IsNegative())
		exit(-1);

	vector<string> lines;
	parseFile(fileName, lines);

	for (int i = 0; i < (int)lines.size(); i += 2) {

		string addr;
		string partialPrivAddr;

		if (lines[i].substr(0, 10) == "Pub Addr: ") {

			addr = lines[i].substr(10);

			switch (addr.data()[0]) {
			case '1':
				addrType = P2PKH; break;
			case '3':
				addrType = P2SH; break;
			case 'b':
			case 'B':
				addrType = BECH32; break;
			default:
				printf("Invalid partialkey info file at line %d\n", i);
				printf("%s Address format not supported\n", addr.c_str());
				continue;
			}

		}
		else {
			printf("[ERROR] Invalid partialkey info file at line %d (\"Pub Addr: \" expected)\n", i);
			exit(-1);
		}

		if (lines[i + 1].substr(0, 13) == "PartialPriv: ") {
			partialPrivAddr = lines[i + 1].substr(13);
		}
		else {
			printf("[ERROR] Invalid partialkey info file at line %d (\"PartialPriv: \" expected)\n", i);
			exit(-1);
		}

		bool partialMode;
		Int partialPrivKey = secp->DecodePrivateKey((char*)partialPrivAddr.c_str(), &partialMode);
		if (privKey.IsNegative()) {
			printf("[ERROR] Invalid partialkey info file at line %d\n", i);
			exit(-1);
		}

		if (partialMode != compressed) {

			printf("[WARNING] Invalid partialkey at line %d (Wrong compression mode, ignoring key)\n", i);
			continue;

		}
		else {

			// Reconstruct the address
			Int fullPriv;
			Point p;
			Int e;
			string cAddr;
			bool found = false;

			// No sym, no endo
			e.Set(&privKey);
			CHECK_ADDR();


			// No sym, endo 1
			e.Set(&privKey);
			e.ModMulK1order(&lambda);
			CHECK_ADDR();

			// No sym, endo 2
			e.Set(&privKey);
			e.ModMulK1order(&lambda2);
			CHECK_ADDR();

			// sym, no endo
			e.Set(&privKey);
			e.Neg();
			e.Add(&secp->order);
			CHECK_ADDR();

			// sym, endo 1
			e.Set(&privKey);
			e.ModMulK1order(&lambda);
			e.Neg();
			e.Add(&secp->order);
			CHECK_ADDR();

			// sym, endo 2
			e.Set(&privKey);
			e.ModMulK1order(&lambda2);
			e.Neg();
			e.Add(&secp->order);
			CHECK_ADDR();

			if (!found) {
				printf("Unable to reconstruct final key from partialkey line %d\n Addr: %s\n PartKey: %s\n",
					i, addr.c_str(), partialPrivAddr.c_str());
			}
		}
	}
}

bool loadBackup(int& idxcount, double& t_Paused, int gpuid) {
	std::string filename = "VSbackup_gpu" + std::to_string(gpuid) + ".dat";
	std::ifstream inFile(filename, std::ios::binary);
	if (inFile) {
		inFile.read(reinterpret_cast<char*>(&idxcount), sizeof(idxcount));
		inFile.read(reinterpret_cast<char*>(&t_Paused), sizeof(t_Paused));
		inFile.close();
		return true;
	}
	else {
		std::cerr << "File not found or error opening file for reading: " << filename << "\n";
		return false;
	}
}

int main(int argc, char* argv[]) {

	std::thread inputThread(monitorKeypress);

	// Global Init
	Timer::Init();

	// Init SecpK1
	Secp256K1* secp = new Secp256K1();
	secp->Init();

	// Browse arguments
	if (argc < 2) {
		fprintf(stderr, "Not enough argument\n");
		printUsage();
		exit(-1);
	}

	int a = 1;
	bool stop = false;
	bool backupMode = false;
	int searchMode = SEARCH_COMPRESSED;
	vector<int> gpuId = { 0 };
	string gpuParsed = "0";
	vector<int> gridSize;
	vector<string> address;
	string outputFile = "";
	uint32_t maxFound = 65536*4;
	int range = 30;
	std::string start = "0";
	int endBits = -1;
	int smMultiplier = 1024; // Default SM multiplier for GPU

	// StringCrack configuration
	StringCrackConfig scConfig;
	memset(&scConfig, 0, sizeof(StringCrackConfig));
	scConfig.enabled = false;
	scConfig.popcountTarget = -1;
	scConfig.popcountMin = 0;
	scConfig.popcountMax = 256;
	scConfig.useRadius = false;
	scConfig.radius = 0;
	scConfig.useRevDoor = false;
	string lockStr = "";
	
	// bitcrack mod
	BITCRACK_PARAM bitcrack, *bc;
	bc = &bitcrack;
	Int maxKey;
		
	maxKey.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
		
	bc->ksStart.SetInt32(1);
	bc->ksNext.Set(&bc->ksStart);
	bc->ksFinish.Set(&maxKey);	

	while (a < argc) {

		if (strcmp(argv[a], "-gpuId") == 0) {
			a++;
			gpuParsed = string(argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-v") == 0) {
			printf("%s\n", RELEASE);
			exit(0);

		}
		else if (strcmp(argv[a], "-o") == 0) {
			a++;
			outputFile = string(argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-start") == 0) {
			a++;
			start = string(argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-center") == 0) {
			a++;
			strncpy(scConfig.centerString, argv[a], 255);
			scConfig.useSEP = true;
			scConfig.enabled = true;
			a++;
		}
		else if (strcmp(argv[a], "-seprange") == 0) {
			a++;
			char* seprange = argv[a];
			char* colon = strchr(seprange, ':');
			if (colon) {
				*colon = '\0';
				scConfig.sepMin = atoi(seprange);
				scConfig.sepMax = atoi(colon + 1);
			} else {
				scConfig.sepMin = 0;
				scConfig.sepMax = atoi(seprange);
			}
			scConfig.useSEP = true;
			scConfig.enabled = true;
			a++;
		}
		else if (strcmp(argv[a], "-radius") == 0) {
			a++;
			scConfig.radius = getInt("radius", argv[a]);
			scConfig.useRadius = true;
			scConfig.useSEP = true;
			scConfig.enabled = true;
			a++;
		}
		else if (strcmp(argv[a], "-radiusrange") == 0) {
			a++;
			string rrStr = string(argv[a]);
			size_t colonPos = rrStr.find(':');
			if (colonPos == string::npos) { fprintf(stderr, "[ERROR] -radiusrange format: min:max\n"); exit(-1); }
			// sepMin stores the lower Hamming bound; radius stores the upper bound.
			// The Gosper engine loop will iterate h = sepMin..radius instead of 0..radius.
			scConfig.sepMin = stoi(rrStr.substr(0, colonPos));
			scConfig.radius = stoi(rrStr.substr(colonPos + 1));
			if (scConfig.sepMin < 0 || scConfig.radius < scConfig.sepMin) {
				fprintf(stderr, "[ERROR] -radiusrange: min must be >= 0 and <= max\n"); exit(-1);
			}
			scConfig.useRadius = true;
			scConfig.useSEP = true;
			scConfig.enabled = true;
			a++;
		}
		else if (strcmp(argv[a], "-revdoor") == 0) {
			scConfig.useRevDoor = true;
			scConfig.useRadius = true;
			scConfig.useSEP = true;
			scConfig.enabled = true;
			a++;
		}
		else if (strcmp(argv[a], "-mitm") == 0) {
			scConfig.useMitm = true;
			scConfig.useRadius = true;
			scConfig.useSEP = true;
			scConfig.enabled = true;
			a++;
		}
		else if (strcmp(argv[a], "-shekinah") == 0) {
			scConfig.useShekinah = true;
			scConfig.useMitm = true;
			scConfig.useRadius = true;
			scConfig.useSEP = true;
			scConfig.enabled = true;
			a++;
		}
		else if (strcmp(argv[a], "-corridorstart") == 0) {
			a++;
			// Parse hex string into 128-bit corridor start
			std::string csHex = std::string(argv[a]);
			// Pad to 32 hex chars (128 bits)
			while (csHex.length() < 32) csHex.insert(0, "0");
			// Parse high and low 64 bits
			std::string hiStr = csHex.substr(0, csHex.length() - 16);
			std::string loStr = csHex.substr(csHex.length() - 16);
			if (hiStr.empty()) hiStr = "0";
			scConfig.shekinahCorridorLo[0] = strtoull(loStr.c_str(), NULL, 16);
			scConfig.shekinahCorridorLo[1] = strtoull(hiStr.c_str(), NULL, 16);
			a++;
		}
		else if (strcmp(argv[a], "-corridorend") == 0) {
			a++;
			std::string ceHex = std::string(argv[a]);
			while (ceHex.length() < 32) ceHex.insert(0, "0");
			std::string hiStr = ceHex.substr(0, ceHex.length() - 16);
			std::string loStr = ceHex.substr(ceHex.length() - 16);
			if (hiStr.empty()) hiStr = "0";
			scConfig.shekinahCorridorHi[0] = strtoull(loStr.c_str(), NULL, 16);
			scConfig.shekinahCorridorHi[1] = strtoull(hiStr.c_str(), NULL, 16);
			a++;
		}
		else if (strcmp(argv[a], "-i") == 0) {
			a++;
			parseFile(string(argv[a]), address);
			a++;

		}
		else if (strcmp(argv[a], "-stop") == 0) {
			stop = true;
			a++;
		}
		else if (strcmp(argv[a], "-random") == 0) {
			randomMode = true;
			a++;
		}
		else if (strcmp(argv[a], "-backup") == 0) {
			backupMode = true;
			a++;
		}
		else if (strcmp(argv[a], "-range") == 0) {
			a++;
			range = (uint64_t)getInt("range", argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-sm") == 0) {
			a++;
			smMultiplier = getInt("sm", argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-end") == 0) {
			a++;
			endBits = getInt("end", argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-m") == 0) {
			a++;
			maxFound = getInt("maxFound", argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-lock") == 0) {
			a++;
			lockStr = string(argv[a]);
			scConfig.enabled = true;
			a++;
		}
		else if (strcmp(argv[a], "-popcount") == 0) {
			a++;
			int pc = getInt("popcount", argv[a]);
			scConfig.popcountTarget = pc;
			scConfig.popcountMin = pc;
			scConfig.popcountMax = pc;
			scConfig.enabled = true;
			a++;
		}
		else if (strcmp(argv[a], "-poprange") == 0) {
			a++;
			string prStr = string(argv[a]);
			size_t colonPos = prStr.find(':');
			if (colonPos == string::npos) { fprintf(stderr, "[ERROR] -poprange format: min:max\n"); exit(-1); }
			scConfig.popcountMin = stoi(prStr.substr(0, colonPos));
			scConfig.popcountMax = stoi(prStr.substr(colonPos + 1));
			scConfig.enabled = true;
			a++;
		}

		else if (a == argc - 1) {
			address.push_back(string(argv[a]));
			a++;
		}
		else {
			printf("Unexpected %s argument\n", argv[a]);
			printUsage();
		}

	}

	fprintf(stdout, "StringCracker v" RELEASE "\n");

	// SEP5: Parse the full comma-separated -gpuId list (e.g. "0,1,2,3")
	gpuId.clear();
	{
		stringstream ss(gpuParsed);
		string token;
		while (getline(ss, token, ',')) {
			// Trim whitespace
			size_t s = token.find_first_not_of(" \t");
			if (s != string::npos) token = token.substr(s);
			size_t e = token.find_last_not_of(" \t");
			if (e != string::npos) token = token.substr(0, e + 1);
			if (!token.empty()) gpuId.push_back(stoi(token));
		}
		if (gpuId.empty()) gpuId.push_back(0);
	}

	if (gridSize.size() == 0) {
		for (int i = 0; i < (int)gpuId.size(); i++) {
			gridSize.push_back(-1);
			gridSize.push_back(128);
		}
	}
	else if (gridSize.size() != gpuId.size() * 2) {
		printf("Invalid gridSize or gpuId argument, must have coherent size\n");
		exit(-1);
	}

	fprintf(stdout, "[SEP5] GPUs requested: %d — IDs:", (int)gpuId.size());
	for (int id : gpuId) fprintf(stdout, " %d", id);
	fprintf(stdout, "\n");

	if (range > 255)
		range = 255;

	Int Range;
	Range.SetInt32(1);

	for (int i = 0; i < range; i++) {
		Range.Mult(2);
	}
	Range.SubOne();

	getKeySpace(string(start + ":+" + Range.GetBase16()), bc, maxKey);
	bc->ksNext.Set(&bc->ksStart);
	checkKeySpace(bc, maxKey);


	// StringCrack: Parse lock string and precompute masks
	if (scConfig.enabled) {
		scConfig.puzzleBits = range;
		if (!lockStr.empty()) parseLockString(lockStr, &scConfig);
		GPUEngine::PrecomputeStringCrackMasks(&scConfig);
		GPUEngine::ComputeBasePoint(secp, &scConfig);
		GPUEngine::ComputeWindowTables(secp, &scConfig);


		// In StringCrack mode: -start is the seed offset, -range is the seed space size
		// -end N limits the scan to a subset (2^endBits seeds from start)
		// Use Int for full 256-bit support
		Int seedOffsetInt;
		seedOffsetInt.SetBase16((char*)start.c_str());
		
		// seedCount = 2^numFreeBits (full space unless range > 256)
		Int seedCountInt;
		seedCountInt.SetInt32(1);
		if (scConfig.numFreeBits < 256) {
			seedCountInt.ShiftL(scConfig.numFreeBits);
		}
		
		// SEP6 Radius Mode: Override seedCount with exact combination count
		if (scConfig.useRadius) {
			if (strlen(scConfig.centerString) == 0) {
				fprintf(stderr, "[ERROR] -radius/-radiusrange requires -center.\n");
				exit(-1);
			}
			if (scConfig.radius > scConfig.numFreeBits) {
				fprintf(stderr, "[ERROR] -radius %d exceeds numFreeBits %d\n", scConfig.radius, scConfig.numFreeBits);
				exit(-1);
			}
			
			// Compute exact combination count: sum(C(n, h) for h=startH..radius)
			// startH is 0 for -radius, or sepMin for -radiusrange.
			int n      = scConfig.numFreeBits;
			int r      = scConfig.radius;
			int startH = (scConfig.sepMin > 0) ? scConfig.sepMin : 0;
			double totalCombinations = 0.0;
			
			double* logFact = new double[n + 1];
			logFact[0] = 0.0;
			for (int i = 1; i <= n; i++) logFact[i] = logFact[i-1] + log((double)i);
			
			for (int h = startH; h <= r; h++) {
				double logC = logFact[n] - logFact[h] - logFact[n - h];
				totalCombinations += exp(logC);
			}
			delete[] logFact;
			
			bool isRange = (scConfig.sepMin > 0);
			printf("\n[SEP6] ============================================\n");
			printf("[SEP6] RADIUS MODE: GPU-native Gosper (combinatorial unranking)\n");
			if (isRange)
				printf("[SEP6] Mode: -radiusrange %d:%d (band search)\n", startH, r);
			else
				printf("[SEP6] Mode: -radius %d (full sphere 0..%d)\n", r, r);
			printf("[SEP6] Center: %s\n", scConfig.centerString);
			printf("[SEP6] Free bits: %d, Hamming layers: h=%d..%d\n", n, startH, r);
			printf("[SEP6] Total combinations: %.0f (~%.2e)\n", totalCombinations, totalCombinations);
			printf("[SEP6] Log2(combinations): %.2f\n", log2(totalCombinations));
			printf("[SEP6] Full space would be: 2^%d = %.2e\n", n, pow(2.0, n));
			printf("[SEP6] Compression ratio: %.4f%% of full space\n", (totalCombinations / pow(2.0, n)) * 100.0);
			printf("[SEP6] ============================================\n\n");
			fflush(stdout);
			
			// For radius mode, seedCount is the total combinations (approximate for stop condition)
			// The actual enumeration is exact via Gosper's hack
			seedCountInt.SetInt32(0);
			// Encode the double as a rough 256-bit integer for stop condition
			// We'll use the actual Gosper state exhaustion as the real stop
			if (totalCombinations < 1.8e19) {
				seedCountInt.bits64[0] = (uint64_t)totalCombinations;
			} else {
				// For very large counts, set high bits too
				seedCountInt.bits64[0] = (uint64_t)fmod(totalCombinations, 1.8446744073709552e19);
				seedCountInt.bits64[1] = (uint64_t)(totalCombinations / 1.8446744073709552e19);
			}
		}
		
		// Calculate end offset based on -end argument
		// If -end is set, limit the scan to 2^endBits from start
		Int seedEndInt;
		seedEndInt.Set(&seedOffsetInt);
		scConfig.endBits = endBits;
		if (endBits > 0 && endBits <= 256) {
			Int endCount;
			endCount.SetInt32(1);
			endCount.ShiftL(endBits);
			seedEndInt.Add(&endCount);
			printf("[StringCrack] Sub-range: 2^%d from offset\n", endBits);
		}
		
		// Store 64-bit truncated versions for GPU
		scConfig.seedOffset = seedOffsetInt.bits64[0];
		scConfig.seedCount = seedCountInt.bits64[0];
		
		// Store full 256-bit versions for CPU calculations
		scConfig.seedOffsetInt.Set(&seedOffsetInt);
		scConfig.seedCountInt.Set(&seedCountInt);
		scConfig.seedEndInt.Set(&seedEndInt);
		
		printf("[StringCrack] Seed offset: %s\n", seedOffsetInt.GetBase16().c_str());
		
		// Print seed count - trim to actual bit length
		int bitLen = seedCountInt.GetBitLength();
		std::string countHex = seedCountInt.GetBase16();
		// Trim leading zeros to match actual bit length
		int expectedHexLen = (bitLen + 3) / 4;  // bits to hex chars
		if ((int)countHex.length() > expectedHexLen) {
			countHex = countHex.substr(countHex.length() - expectedHexLen);
		}
		printf("[StringCrack] Seed count:  %s (2^%d)\n",
				countHex.c_str(), scConfig.numFreeBits);

		// ═══ SHEKINAH MATRIX VALIDATION ═══
		if (scConfig.useShekinah) {
			if (scConfig.shekinahCorridorLo[0] == 0 && scConfig.shekinahCorridorLo[1] == 0) {
				fprintf(stderr, "[ERROR] -shekinah requires -corridorstart.\n");
				exit(-1);
			}
			if (scConfig.shekinahCorridorHi[0] == 0 && scConfig.shekinahCorridorHi[1] == 0) {
				fprintf(stderr, "[ERROR] -shekinah requires -corridorend.\n");
				exit(-1);
			}
			printf("\n[SEPHOLY] ============================================\n");
			printf("[SEPHOLY] SHEKINAH MATRIX MODE ENABLED\n");
			printf("[SEPHOLY] Corridor Start: %016llX%016llX\n",
				(unsigned long long)scConfig.shekinahCorridorLo[1],
				(unsigned long long)scConfig.shekinahCorridorLo[0]);
			printf("[SEPHOLY] Corridor End:   %016llX%016llX\n",
				(unsigned long long)scConfig.shekinahCorridorHi[1],
				(unsigned long long)scConfig.shekinahCorridorHi[0]);
			printf("[SEPHOLY] Puzzle bits:    %d\n", scConfig.puzzleBits);
			printf("[SEPHOLY] Popcount range: %d:%d\n", scConfig.popcountMin, scConfig.popcountMax);
			printf("[SEPHOLY] ============================================\n\n");
			fflush(stdout);
		}
	}

	{

		fprintf(stdout, "[keyspace]  range=2^%d\n", range);
		fprintf(stdout, "[keyspace]  start=%s\n", bc->ksStart.GetBase16().c_str());
		
		// If -end is specified, use seedEndInt for the keyspace end
		if (scConfig.enabled && endBits > 0) {
			fprintf(stdout, "[keyspace]    end=%s\n", scConfig.seedEndInt.GetBase16().c_str());
		} else {
			fprintf(stdout, "[keyspace]    end=%s\n", bc->ksFinish.GetBase16().c_str());
		}
		
		if (randomMode) fprintf(stdout, "Random Mode Enabled !\n");
		if (scConfig.enabled) {
			fprintf(stdout, "[StringCrack] Mode ENABLED\n");
			fprintf(stdout, "[StringCrack] -start = seed offset 0x%s\n",
				scConfig.seedOffsetInt.GetBase16().c_str());
			fprintf(stdout, "[StringCrack] -range = 2^%d seed space\n", scConfig.numFreeBits);
		}
		fflush(stdout);


		idxcount = 0;
		t_Paused = 0;
		Pause = false;

		if (backupMode) {
			loadBackup(idxcount, t_Paused, gpuId[0]);
			fprintf(stdout, "Backup enabled ! \n");
		}
	repeatP:
		Paused = false;
		VanitySearch* v = new VanitySearch(secp, address, searchMode, stop, outputFile, maxFound, bc,
			scConfig.enabled ? &scConfig : NULL);
		v->smMultiplier = smMultiplier;
		v->Search(gpuId, gridSize);

		while (Paused) {
			Timer::SleepMillis(100);
			if (!Pause) {
				fprintf(stdout, "\nResuming...\n");
				goto repeatP;
				
			}
		}
	}

	stopMonitorKey = true;

	Timer::SleepMillis(100);

	if (inputThread.joinable()) {
		inputThread.join();
	}

	return 0;
}
