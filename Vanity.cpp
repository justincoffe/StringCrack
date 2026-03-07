/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Vanity.h"
#include "Base58.h"
#include "Bech32.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Wildcard.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <thread>
#include <atomic>
#include <ctime> 
#include <iostream>
#include <fstream>



//#define GRP_SIZE 256

using namespace std;

//Point Gn[GRP_SIZE / 2];
//Point _2Gn;

VanitySearch::VanitySearch(Secp256K1* secp, vector<std::string>& inputAddresses, int searchMode,
	bool stop, string outputFile, uint32_t maxFound, BITCRACK_PARAM* bc,
	StringCrackConfig* scConfig):inputAddresses(inputAddresses) 
{
	this->secp = secp;
	this->searchMode = searchMode;
	this->stopWhenFound = stop;
	this->outputFile = outputFile;
	this->numGPUs = 0;
	this->maxFound = maxFound;	
	this->searchType = -1;
	this->bc = bc;
	this->scConfig = scConfig;

	rseed(static_cast<unsigned long>(time(NULL)));
	
	addresses.clear();

	// Create a 65536 items lookup table
	ADDRESS_TABLE_ITEM t;
	t.found = true;
	t.items = NULL;
	for (int i = 0; i < 65536; i++)
		addresses.push_back(t);
	
	// Insert addresses
	bool loadingProgress = (inputAddresses.size() > 1000);
	if (loadingProgress)
		fprintf(stdout, "[Building lookup16   0.0%%]\r");

	nbAddress = 0;
	onlyFull = true;

	for (int i = 0; i < (int)inputAddresses.size(); i++) 
	{
		ADDRESS_ITEM it;
		std::vector<ADDRESS_ITEM> itAddresses;

		if (initAddress(inputAddresses[i], &it)) {
			bool* found = new bool;
			*found = false;
			it.found = found;
			itAddresses.push_back(it);
		}

		if (itAddresses.size() > 0) 
		{
			// Add the item to all correspoding addresses in the lookup table
			for (int j = 0; j < (int)itAddresses.size(); j++) 
			{
				address_t p = itAddresses[j].sAddress;

				if (addresses[p].items == NULL) {
					addresses[p].items = new vector<ADDRESS_ITEM>();
					addresses[p].found = false;
					usedAddress.push_back(p);
				}
				(*addresses[p].items).push_back(itAddresses[j]);
			}
			onlyFull &= it.isFull;
			nbAddress++;
		}

		if (loadingProgress && i % 1000 == 0)
			fprintf(stdout, "[Building lookup16 %5.1f%%]\r", (((double)i) / (double)(inputAddresses.size() - 1)) * 100.0);
	}

	if (loadingProgress)
		fprintf(stdout, "\n");

	if (nbAddress == 0) 
	{
		fprintf(stderr, "[ERROR] VanitySearch: nothing to search !\n");
		exit(-1);
	}

	// Second level lookup
	uint32_t unique_sAddress = 0;
	uint32_t minI = 0xFFFFFFFF;
	uint32_t maxI = 0;
	for (int i = 0; i < (int)addresses.size(); i++) 
	{
		
		if (addresses[i].items) 
		{
			
			LADDRESS lit;
			lit.sAddress = i;
			if (addresses[i].items) 
			{
				for (int j = 0; j < (int)addresses[i].items->size(); j++) 
				{
					lit.lAddresses.push_back((*addresses[i].items)[j].lAddress);
					
				}
			}

			sort(lit.lAddresses.begin(), lit.lAddresses.end());
			usedAddressL.push_back(lit);
			if ((uint32_t)lit.lAddresses.size() > maxI) maxI = (uint32_t)lit.lAddresses.size();
			if ((uint32_t)lit.lAddresses.size() < minI) minI = (uint32_t)lit.lAddresses.size();
			unique_sAddress++;
		}

		if (loadingProgress)
			fprintf(stdout, "[Building lookup32 %.1f%%]\r", ((double)i * 100.0) / (double)addresses.size());
	}

	if (loadingProgress)
		fprintf(stdout, "\n");
	
	string searchInfo = string(searchModes[searchMode]);
	if (nbAddress < 10) 
	{	
		for (size_t i = 0; i < nbAddress; i++)
		{
			fprintf(stdout, "Search: %s [%s]\n", inputAddresses[i].c_str(), searchInfo.c_str());
		}
	}
	else 
	{		
		fprintf(stdout, "Search: %d (Lookup size %d,[%d,%d]) [%s]\n", nbAddress, unique_sAddress, minI, maxI, searchInfo.c_str());
	}

	//// Compute Generator table G[n] = (n+1)*G
	//Point g = secp->G;
	//Gn[0] = g;
	//g = secp->DoubleDirect(g);
	//Gn[1] = g;
	//for (int i = 2; i < GRP_SIZE / 2; i++) {
	//	g = secp->AddDirect(g, secp->G);
	//	Gn[i] = g;
	//}
	//// _2Gn = CPU_GRP_SIZE*G
	//_2Gn = secp->DoubleDirect(Gn[GRP_SIZE / 2 - 1]);

	// Constant for endomorphism
	// if a is a nth primitive root of unity, a^-1 is also a nth primitive root.
	// beta^3 = 1 mod p implies also beta^2 = beta^-1 mop (by multiplying both side by beta^-1)
	// (beta^3 = 1 mod p),  beta2 = beta^-1 = beta^2
	// (lambda^3 = 1 mod n), lamba2 = lamba^-1 = lamba^2
	beta.SetBase16("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
	lambda.SetBase16("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
	beta2.SetBase16("851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40");
	lambda2.SetBase16("ac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce");

	startKey.Set(&bc->ksNext);	

	char* ctimeBuff;
	time_t now = time(NULL);
	ctimeBuff = ctime(&now);
	fprintf(stdout, "Current task START time: %s", ctimeBuff);
	fflush(stdout);
}

bool VanitySearch::isSingularAddress(std::string pref) {

	// check is the given address contains only 1
	bool only1 = true;
	int i = 0;
	while (only1 && i < (int)pref.length()) {
		only1 = pref.data()[i] == '1';
		i++;
	}
	return only1;
}

bool VanitySearch::initAddress(std::string& address, ADDRESS_ITEM* it) {

	std::vector<unsigned char> result;
	string dummy1 = address;
	int nbDigit = 0;
	bool wrong = false;

	if (address.length() < 2) {
		fprintf(stdout, "Ignoring address \"%s\" (too short)\n", address.c_str());
		return false;
	}

	int aType = -1;

	switch (address.data()[0]) {
	case '1':
		aType = P2PKH;
		break;
	case '3':
		aType = P2SH;
		break;
	case 'b':
	case 'B':
		std::transform(address.begin(), address.end(), address.begin(), ::tolower);
		if (strncmp(address.c_str(), "bc1q", 4) == 0)
			aType = BECH32;
		break;
	}

	if (aType == -1) {
		fprintf(stdout, "Ignoring address \"%s\" (must start with 1 or 3 or bc1q)\n", address.c_str());
		return false;
	}

	if (searchType == -1) searchType = aType;
	if (aType != searchType) {
		fprintf(stdout, "Ignoring address \"%s\" (P2PKH, P2SH or BECH32 allowed at once)\n", address.c_str());
		return false;
	}

	if (aType == BECH32) {

		// BECH32
		uint8_t witprog[40];
		size_t witprog_len;
		int witver;
		const char* hrp = "bc";

		int ret = segwit_addr_decode(&witver, witprog, &witprog_len, hrp, address.c_str());

		// Try to attack a full address ?
		if (ret && witprog_len == 20) {
						
			it->isFull = true;
			memcpy(it->hash160, witprog, 20);
			it->sAddress = *(address_t*)(it->hash160);
			it->lAddress = *(addressl_t*)(it->hash160);
			it->address = (char*)address.c_str();
			it->addressLength = (int)address.length();
			return true;

		}

		if (address.length() < 5) {
			fprintf(stdout, "Ignoring address \"%s\" (too short, length<5 )\n", address.c_str());
			return false;
		}

		if (address.length() >= 36) {
			fprintf(stdout, "Ignoring address \"%s\" (too long, length>36 )\n", address.c_str());
			return false;
		}

		uint8_t data[64];
		memset(data, 0, 64);
		size_t data_length;
		if (!bech32_decode_nocheck(data, &data_length, address.c_str() + 4)) {
			fprintf(stdout, "Ignoring address \"%s\" (Only \"023456789acdefghjklmnpqrstuvwxyz\" allowed)\n", address.c_str());
			return false;
		}
		
		it->sAddress = *(address_t*)data;		
		it->isFull = false;
		it->lAddress = 0;
		it->address = (char*)address.c_str();
		it->addressLength = (int)address.length();

		return true;
	}
	else {

		// P2PKH/P2SH
		wrong = !DecodeBase58(address, result);

		if (wrong) {
			fprintf(stdout, "Ignoring address \"%s\" (0, I, O and l not allowed)\n", address.c_str());
			return false;
		}

		// Try to attack a full address ?
		if (result.size() > 21) {
			
			it->isFull = true;
			memcpy(it->hash160, result.data() + 1, 20);
			it->sAddress = *(address_t*)(it->hash160);
			it->lAddress = *(addressl_t*)(it->hash160);
			it->address = (char*)address.c_str();
			it->addressLength = (int)address.length();
			return true;
		}

		// Address containing only '1'
		if (isSingularAddress(address)) {

			if (address.length() > 21) {
				fprintf(stdout, "Ignoring address \"%s\" (Too much 1)\n", address.c_str());
				return false;
			}
			
			it->isFull = false;
			it->sAddress = 0;
			it->lAddress = 0;
			it->address = (char*)address.c_str();
			it->addressLength = (int)address.length();
			return true;
		}

		// Search for highest hash160 16bit address (most probable)
		while (result.size() < 25) {
			DecodeBase58(dummy1, result);
			if (result.size() < 25) {
				dummy1.append("1");
				nbDigit++;
			}
		}

		if (searchType == P2SH) {
			if (result.data()[0] != 5) {
				fprintf(stdout, "Ignoring address \"%s\" (Unreachable, 31h1 to 3R2c only)\n", address.c_str());
				return false;
			}
		}

		if (result.size() != 25) {
			fprintf(stdout, "Ignoring address \"%s\" (Invalid size)\n", address.c_str());
			return false;
		}

		it->sAddress = *(address_t*)(result.data() + 1);

		dummy1.append("1");
		DecodeBase58(dummy1, result);

		if (result.size() == 25) {
			it->sAddress = *(address_t*)(result.data() + 1);
			nbDigit++;
		}
		
		it->isFull = false;
		it->lAddress = 0;
		it->address = (char*)address.c_str();
		it->addressLength = (int)address.length();

		return true;
	}
}

void VanitySearch::enumCaseUnsentiveAddress(std::string s, std::vector<std::string>& list) {

	char letter[64];
	int letterpos[64];
	int nbLetter = 0;
	int length = (int)s.length();

	for (int i = 1; i < length; i++) {
		char c = s.data()[i];
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
			letter[nbLetter] = tolower(c);
			letterpos[nbLetter] = i;
			nbLetter++;
		}
	}

	int total = 1 << nbLetter;

	for (int i = 0; i < total; i++) {

		char tmp[64];
		strcpy(tmp, s.c_str());

		for (int j = 0; j < nbLetter; j++) {
			int mask = 1 << j;
			if (mask & i) tmp[letterpos[j]] = toupper(letter[j]);
			else         tmp[letterpos[j]] = letter[j];
		}

		list.push_back(string(tmp));

	}

}

// ----------------------------------------------------------------------------

void VanitySearch::output(string addr, string pAddr, string pAddrHex, std::string pubKey) {

#ifdef WIN64
	WaitForSingleObject(ghMutex, INFINITE);
#else
	pthread_mutex_lock(&ghMutex);
#endif

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




	if (f != stdout)
		fprintf(f, "\nPublic Addr: %s\n", addr.c_str());	
	fprintf(stdout, "\nPublic Addr: %s\n", addr.c_str());

	switch (searchType) {
	case P2PKH:
		if (f != stdout)
			fprintf(f, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
		fprintf(stdout, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
		break;
	case P2SH:
		if (f != stdout)
			fprintf(f, "Priv (WIF): p2wpkh-p2sh:%s\n", pAddr.c_str());
		fprintf(stdout, "Priv (WIF): p2wpkh-p2sh:%s\n", pAddr.c_str());
		break;
	case BECH32:
		if (f != stdout)
			fprintf(f, "Priv (WIF): p2wpkh:%s\n", pAddr.c_str());
		fprintf(stdout, "Priv (WIF): p2wpkh:%s\n", pAddr.c_str());
		break;
	}

	if (f != stdout)
		fprintf(f, "Priv (HEX): 0x%064s\n", pAddrHex.c_str());	
	fprintf(stdout, "Priv (HEX): 0x%064s\n", pAddrHex.c_str());
	fprintf(stdout, "\n");

	if (f != stdout)
		fflush(f);
	fflush(stdout);
	//fflush(stderr);	

	if (needToClose)
		fclose(f);

#ifdef WIN64
	ReleaseMutex(ghMutex);
#else
	pthread_mutex_unlock(&ghMutex);
#endif
}

void VanitySearch::updateFound() {

	// Check if all addresses has been found
	// Needed only if stopWhenFound is asked
	if (stopWhenFound) 	{

		bool allFound = true;
		for (int i = 0; i < (int)usedAddress.size(); i++) {
			bool iFound = true;
			address_t p = usedAddress[i];
			if (!addresses[p].found) {
				if (addresses[p].items) {
					for (int j = 0; j < (int)addresses[p].items->size(); j++) {
						iFound &= *((*addresses[p].items)[j].found);
					}
				}
				addresses[usedAddress[i]].found = iFound;
			}
			allFound &= iFound;
		}

		endOfSearch = allFound;		
	}		
}

bool VanitySearch::checkPrivKey(string addr, Int& key, int32_t incr, int endomorphism, bool mode) {

	Int k(&key);	

	if (incr < 0) {
		k.Add((uint64_t)(-incr));
		k.Neg();
		k.Add(&secp->order);		
	}
	else {
		k.Add((uint64_t)incr);
	}

	// Endomorphisms
	switch (endomorphism) {
	case 1:
		k.ModMulK1order(&lambda);		
		break;
	case 2:
		k.ModMulK1order(&lambda2);		
		break;
	}

	// Check addresses
	Point p = secp->ComputePublicKey(&k);	

	string chkAddr = secp->GetAddress(searchType, mode, p);
	if (chkAddr != addr) {

		// Key may be the opposite one (negative zero or compressed key)
		k.Neg();
		k.Add(&secp->order);
		p = secp->ComputePublicKey(&k);
		
		string chkAddr = secp->GetAddress(searchType, mode, p);
		if (chkAddr != addr) {
			fprintf(stdout, "\nWarning, wrong private key generated !\n");
			fprintf(stdout, "  Addr :%s\n", addr.c_str());
			fprintf(stdout, "  Check:%s\n", chkAddr.c_str());
			fprintf(stdout, "  Endo:%d incr:%d comp:%d\n", endomorphism, incr, mode);
			return false;
		}

	}

	output(addr, secp->GetPrivAddress(mode, k), k.GetBase16(), secp->GetPublicKeyHex(mode, p));

	return true;
}

void VanitySearch::checkAddrSSE(uint8_t* h1, uint8_t* h2, uint8_t* h3, uint8_t* h4,
	int32_t incr1, int32_t incr2, int32_t incr3, int32_t incr4,
	Int& key, int endomorphism, bool mode) {

	vector<string> addr = secp->GetAddress(searchType, mode, h1, h2, h3, h4);

	for (int i = 0; i < (int)inputAddresses.size(); i++) {

		if (Wildcard::match(addr[0].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[0], key, incr1, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}

		if (Wildcard::match(addr[1].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[1], key, incr2, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}

		if (Wildcard::match(addr[2].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[2], key, incr3, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}

		if (Wildcard::match(addr[3].c_str(), inputAddresses[i].c_str())) {

			// Found it !      
			if (checkPrivKey(addr[3], key, incr4, endomorphism, mode)) {
				nbFoundKey++;
				//patternFound[i] = true;
				updateFound();
			}
		}
	}
}

void VanitySearch::checkAddr(int prefIdx, uint8_t* hash160, Int& key, int32_t incr, int endomorphism, bool mode) {
	
	vector<ADDRESS_ITEM>* pi = addresses[prefIdx].items;	


	if (onlyFull) {

		// Full addresses
		for (int i = 0; i < (int)pi->size(); i++) {

			if (stopWhenFound && *((*pi)[i].found))
				continue;

			if (ripemd160_comp_hash((*pi)[i].hash160, hash160)) {

				// Found it !
				*((*pi)[i].found) = true;
				// You believe it ?
				if (checkPrivKey(secp->GetAddress(searchType, mode, hash160), key, incr, endomorphism, mode)) {
					nbFoundKey++;
					updateFound();
				}

			}

		}

	}
	else {
		char a[64];

		string addr = secp->GetAddress(searchType, mode, hash160);

		for (int i = 0; i < (int)pi->size(); i++) {

			if (stopWhenFound && *((*pi)[i].found))
				continue;

			strncpy(a, addr.c_str(), (*pi)[i].addressLength);
			a[(*pi)[i].addressLength] = 0;

			if (strcmp((*pi)[i].address, a) == 0) {

				// Found it !
				*((*pi)[i].found) = true;
				if (checkPrivKey(addr, key, incr, endomorphism, mode)) {
					nbFoundKey++;
					updateFound();
				}

			}

		}

	}


}

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam) {
#else
void* _FindKeyGPU(void* lpParam) {
#endif
	TH_PARAM* p = (TH_PARAM*)lpParam;
	p->obj->FindKeyGPU(p);
	return 0;
}

void VanitySearch::checkAddresses(bool compressed, Int key, int i, Point p1) {

	unsigned char h0[20];
	Point pte1[1];
	Point pte2[1];

	// Point
	secp->GetHash160(searchType, compressed, p1, h0);
	address_t pr0 = *(address_t*)h0;
	if (addresses[pr0].items)
		checkAddr(pr0, h0, key, i, 0, compressed);	
}

void VanitySearch::checkAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4) {

	unsigned char h0[20];
	unsigned char h1[20];
	unsigned char h2[20];
	unsigned char h3[20];
	Point pte1[4];
	Point pte2[4];
	address_t pr0;
	address_t pr1;
	address_t pr2;
	address_t pr3;

	// Point -------------------------------------------------------------------------
	secp->GetHash160(searchType, compressed, p1, p2, p3, p4, h0, h1, h2, h3);	

	pr0 = *(address_t*)h0;
	pr1 = *(address_t*)h1;
	pr2 = *(address_t*)h2;
	pr3 = *(address_t*)h3;

	if (addresses[pr0].items)
		checkAddr(pr0, h0, key, i, 0, compressed);
	if (addresses[pr1].items)
		checkAddr(pr1, h1, key, i + 1, 0, compressed);
	if (addresses[pr2].items)
		checkAddr(pr2, h2, key, i + 2, 0, compressed);
	if (addresses[pr3].items)
		checkAddr(pr3, h3, key, i + 3, 0, compressed);	
}

void VanitySearch::getGPUStartingKeys(Int& tRangeStart, Int& tRangeEnd, int groupSize, int nbThread, Point *p, uint64_t Progress) {
		
	uint32_t grp_startkeys = nbThread/256;

	//New setting key by fixedpaul using addition on secp with batch modular inverse, super fast, multithreading not needed

	Int stepThread;
	Int numthread;

	stepThread.Set(&tRangeEnd);
	stepThread.Sub(&tRangeStart);
	stepThread.AddOne();
	numthread.SetInt32(nbThread);
	stepThread.Div(&numthread);

	Point Pdouble;
	Int kDouble;

	kDouble.Set(&stepThread);
	kDouble.Mult(grp_startkeys);
	Pdouble = secp->ComputePublicKey(&kDouble);

	Point P_start;
	Int kStart;

	kStart.Set(&stepThread);
	kStart.Mult(grp_startkeys / 2);
	kStart.Add(groupSize / 2 + Progress);

	

	P_start = secp->ComputePublicKey(&kStart);

	p[grp_startkeys / 2] = secp->ComputePublicKey(&tRangeStart);
	p[grp_startkeys / 2] = secp->AddDirect(p[grp_startkeys / 2], P_start);


	Int key_delta;
	Point* p_delta;
	p_delta = new Point[grp_startkeys / 2];

	key_delta.Set(&stepThread);

	

	p_delta[0] = secp->ComputePublicKey(&key_delta);
	key_delta.Add(&stepThread);
	p_delta[1] = secp->ComputePublicKey(&key_delta);

	for (size_t i = 2; i < grp_startkeys / 2; i++) {
		p_delta[i] = secp->AddDirect(p_delta[i - 1], p_delta[0]);
	}

	Int* dx;
	Int* subp;

	subp = new Int[grp_startkeys / 2 + 1];
	dx = new Int[grp_startkeys / 2 + 1];

	uint32_t j;
	//uint32_t i;

	for (size_t i = grp_startkeys / 2; i < nbThread; i += grp_startkeys) {

		double percentage = (100.0 * (double)(i + grp_startkeys / 2)) / (double)(nbThread);
		printf("Setting starting keys... [%.2f%%] \r", percentage);
		fflush(stdout);


		for (j = 0; j < grp_startkeys / 2; j++) {
			dx[j].ModSub(&p_delta[j].x, &p[i].x);
		}
		dx[grp_startkeys / 2].ModSub(&Pdouble.x, &p[i].x);

		Int newValue;
		Int inverse;

		subp[0].Set(&dx[0]);
		for (size_t j = 1; j < grp_startkeys / 2 + 1; j++) {
			subp[j].ModMulK1(&subp[j - 1], &dx[j]);
		}

		// Do the inversion
		inverse.Set(&subp[grp_startkeys / 2]);
		inverse.ModInv();

		for (j = grp_startkeys / 2; j > 0; j--) {
			newValue.ModMulK1(&subp[j - 1], &inverse);
			inverse.ModMulK1(&dx[j]);
			dx[j].Set(&newValue);
		}

		dx[0].Set(&inverse);

		Int _s;
		Int _p;
		Int dy;
		Int syn;
		syn.Set(&p[i].y);
		syn.ModNeg();



		for (j = 0; j < grp_startkeys / 2 - 1; j++) {

			dy.ModSub(&p_delta[j].y, &p[i].y);
			_s.ModMulK1(&dy, &dx[j]);

			_p.ModSquareK1(&_s);

			p[i + j + 1].x.ModSub(&_p, &p[i].x);
			p[i + j + 1].x.ModSub(&p_delta[j].x);

			p[i + j + 1].y.ModSub(&p_delta[j].x, &p[i + j + 1].x);
			p[i + j + 1].y.ModMulK1(&_s);
			p[i + j + 1].y.ModSub(&p_delta[j].y);

			dy.ModSub(&syn, &p_delta[j].y);
			_s.ModMulK1(&dy, &dx[j]);

			_p.ModSquareK1(&_s);

			p[i - j - 1].x.ModSub(&_p, &p[i].x);
			p[i - j - 1].x.ModSub(&p_delta[j].x);

			p[i - j - 1].y.ModSub(&p[i - j - 1].x, &p_delta[j].x);
			p[i - j - 1].y.ModMulK1(&_s);
			p[i - j - 1].y.ModSub(&p_delta[j].y, &p[i - j - 1].y);
		}

		dy.ModSub(&syn, &p_delta[j].y);
		_s.ModMulK1(&dy, &dx[j]);

		_p.ModSquareK1(&_s);


		p[i - j - 1].x.ModSub(&_p, &p[i].x);
		p[i - j - 1].x.ModSub(&p_delta[j].x);

		p[i - j - 1].y.ModSub(&p[i - j - 1].x, &p_delta[j].x);
		p[i - j - 1].y.ModMulK1(&_s);
		p[i - j - 1].y.ModSub(&p_delta[j].y, &p[i - j - 1].y);

		if (i + grp_startkeys < nbThread) {

			dy.ModSub(&Pdouble.y, &p[i].y);
			_s.ModMulK1(&dy, &dx[grp_startkeys / 2]);

			_p.ModSquareK1(&_s);

			p[i + grp_startkeys].x.ModSub(&_p, &p[i].x);
			p[i + grp_startkeys].x.ModSub(&Pdouble.x);

			p[i + grp_startkeys].y.ModSub(&Pdouble.x, &p[i + grp_startkeys].x);
			p[i + grp_startkeys].y.ModMulK1(&_s);
			p[i + grp_startkeys].y.ModSub(&Pdouble.y);
		}
	}

	delete[] subp;
	delete[] dx;
	delete[] p_delta;
}

void VanitySearch::FindKeyGPU(TH_PARAM* ph) {

	bool ok = true;
	double t0;
	double ttot;
	uint64_t keys_n = 0;
	static uint64_t keys_n_prev = 0;
	static double tprev = 0.0;

	// Global init
	int thId = ph->threadId;

	// FIX: Make keyspace bounds completely thread-local
	Int local_ksStart;
	local_ksStart.Set(&bc->ksStart);
	Int local_ksFinish;
	local_ksFinish.Set(&bc->ksFinish);

	// FIX: Localize the iteration counter so threads don't corrupt each other
	int local_idxcount = 0;

	GPUEngine g(ph->gpuId, maxFound, ph->smMultiplier);
	int numThreadsGPU = g.GetNbThread();
	int STEP_SIZE = g.GetStepSize();
	Point* publicKeys = new Point[numThreadsGPU];
	vector<ITEM> found;

	Point RandomJump_P;
	Int RandomJump_K;
	Int RandomJump_K_last;
	Int RandomJump_K_tot;
	RandomJump_K.SetInt32(STEP_SIZE);
	RandomJump_K_last.SetInt32(0);
	RandomJump_K_tot.SetInt32(0);

	fprintf(stdout, "GPU: %s\n", g.deviceName.c_str());
	fflush(stdout);
	counters[thId] = 0;
	
	g.SetSearchMode(searchMode);
	g.SetSearchType(searchType);
	if (onlyFull) {
		g.SetAddress(usedAddressL, nbAddress);
	} else {
		g.SetAddress(usedAddress);
	}

	bool useStringCrack = (scConfig != NULL && scConfig->enabled);

	// SEP dual-filter tracking variables
	int upper_hd = 0;
	int upper_abs_pop = 0;

	Int stepThread;
	Int taskSize;
	Int numthread;
	numthread.SetInt32(numThreadsGPU);

	Int privkey;
	Int part_key;
	Int keycount;

	t0 = Timer::get_tick();
	endOfSearch = false;

	// Thread-safe tracking variables
	Int sc_currentSeed;
	Int sc_limitSeed;
	Int sc_blockEndSeed; // NEW: Tracks the exact end of the block
	int sc_lowerFreeBitsCount = 0;
	bool needsNewBlock = true;
	uint64_t sc_keys_n = 0;
	static uint64_t sc_keys_n_prev = 0;
	static double sc_last_print_time = 0.0; // FIX: Track exact time of last print
	
	Int previous_ksStart; // NEW: Local delta tracking
	bool isFirstBlock = true; // NEW: Local init flag

	// ==============================================================
	// 256-BIT UNIFIED AUTO-HD & CAPPED WEAK BITS MUTATION SETUP
	// ==============================================================
	struct XorMask256 {
		uint64_t m[4] = {0, 0, 0, 0};
		// Natively comparable so std::sort and std::unique can deduplicate
		bool operator<(const XorMask256& o) const {
			for (int i = 3; i >= 0; i--) {
				if (m[i] != o.m[i]) return m[i] < o.m[i];
			}
			return false;
		}
		bool operator==(const XorMask256& o) const {
			return m[0] == o.m[0] && m[1] == o.m[1] && m[2] == o.m[2] && m[3] == o.m[3];
		}
	};

	int current_mutation = 0;
	std::vector<XorMask256> xor_masks;
	
	XorMask256 zero_mask;
	xor_masks.push_back(zero_mask); // HD 0 (The exact AI prediction)
	
	if (scConfig != NULL) {
		
		// 1. AUTO-HD LOGIC (Geographically constrained to < 64 to prevent massive explosions)
		if (scConfig->numLockedBits > 0 && scConfig->autoHD > 0) {
			// HD 1
			if (scConfig->autoHD >= 1) {
				for(int i = 0; i < scConfig->numLockedBits; i++) {
					int pos1 = scConfig->lockedBits[i].position;
					if (pos1 < 64) {
						XorMask256 mask;
						mask.m[pos1 >> 6] |= (1ULL << (pos1 & 63));
						xor_masks.push_back(mask); 
					}
				}
			}
			// HD 2
			if (scConfig->autoHD >= 2) {
				for(int i = 0; i < scConfig->numLockedBits; i++) {
					for(int j = i + 1; j < scConfig->numLockedBits; j++) {
						int pos1 = scConfig->lockedBits[i].position;
						int pos2 = scConfig->lockedBits[j].position;
						if (pos1 < 64 && pos2 < 64) {
							XorMask256 mask;
							mask.m[pos1 >> 6] |= (1ULL << (pos1 & 63));
							mask.m[pos2 >> 6] |= (1ULL << (pos2 & 63));
							xor_masks.push_back(mask); 
						}
					}
				}
			}
		}

		// 2. WEAK BITS LOGIC (Unleashed across the entire 256-bit space)
		if (scConfig->numWeakBits > 0) {
			std::vector<int> wBits;
			// NO MORE 64-BIT LIMIT! Weak bits can be placed anywhere up to 255
			for (int i = 0; i < scConfig->numWeakBits; i++) {
				if (scConfig->weakBits[i] < 256) wBits.push_back(scConfig->weakBits[i]);
			}
			
			int nW = wBits.size();
			int max_weak_hd = scConfig->weakMaxHD; // Now pulls dynamically from CLI

			int total_combinations = 1 << nW; 
			for (int mask = 1; mask < total_combinations; mask++) {
				
				int current_hd = 0;
				int temp_mask = mask;
				while (temp_mask > 0) {
					current_hd += (temp_mask & 1);
					temp_mask >>= 1;
				}

				if (current_hd <= max_weak_hd) {
					XorMask256 current_xor;
					for (int b = 0; b < nW; b++) {
						if ((mask >> b) & 1) {
							int pos = wBits[b];
							current_xor.m[pos >> 6] |= (1ULL << (pos & 63));
						}
					}
					xor_masks.push_back(current_xor);
				}
			}
		}

		// 3. ZERO OVERHEAD GUARANTEE (Remove duplicate realities)
		std::sort(xor_masks.begin(), xor_masks.end());
		xor_masks.erase(std::unique(xor_masks.begin(), xor_masks.end()), xor_masks.end());
	}
	// ==============================================================

	// Thread-local seed variables for Multi-GPU support
	Int thread_currentSeed;
	Int thread_limitSeed;
	Int thread_blockEndSeed;

	if (useStringCrack) {
		printf("[Hybrid Engine] Initializing CPU-GPU Workload Split...\n");
		
		// CPU detects the contiguous block of lower free bits for the GPU
		while (sc_lowerFreeBitsCount < scConfig->numFreeBits &&
			   scConfig->freeBitPositions[sc_lowerFreeBitsCount] == sc_lowerFreeBitsCount) {
			sc_lowerFreeBitsCount++;
		}

		if (sc_lowerFreeBitsCount < 20) {
			printf("[Hybrid Engine] WARNING: Low contiguous free bits (%d). CPU bottleneck highly likely on massive arrays.\n", sc_lowerFreeBitsCount);
		}

		printf("[Hybrid Engine] Muscle: %d contiguous lower free bits (Block size: 2^%d).\n", sc_lowerFreeBitsCount, sc_lowerFreeBitsCount);
		
		Int globalStart, globalEnd, totalSpace, chunkSpace, myStart, myEnd;
		globalStart.Set(&scConfig->seedOffsetInt);
		
		if (scConfig->endBits > 0) {
			globalEnd.Set(&scConfig->seedEndInt);
		} else {
			globalEnd.Set(&scConfig->seedCountInt);
		}

		// Calculate total seeds and divide by number of GPUs
		totalSpace.Set(&globalEnd);
		totalSpace.Sub(&globalStart);
		
		Int gpusInt;
		gpusInt.SetInt32(numGPUs);
		chunkSpace.Set(&totalSpace);
		chunkSpace.Div(&gpusInt);

		// Calculate this specific GPU's start seed: offset + (thId * chunk)
		myStart.Set(&chunkSpace);
		myStart.Mult(thId);
		myStart.Add(&globalStart);

		// Calculate this specific GPU's end seed
		if (thId == numGPUs - 1) {
			// The last GPU always takes the exact remainder to the end
			myEnd.Set(&globalEnd); 
		} else {
			myEnd.Set(&myStart);
			myEnd.Add(&chunkSpace);
		}

		// Isolate the seed states to the local thread context
		// (Variables are now declared at function level)
		
		thread_currentSeed.Set(&myStart);
		thread_limitSeed.Set(&myEnd);
		
		printf("[Hybrid Engine] GPU %d Workload: Seeds %s to %s\n", thId, myStart.GetBase16().c_str(), myEnd.GetBase16().c_str());
	} else {
		// Normal Mode Setup - WITH MULTI-GPU WORKLOAD SPLIT
		// This block is ONLY executed if -lock is NOT provided. StringCrack is safe.
		Int globalStart, globalEnd, totalSpace, chunkSpace;
			globalStart.Set(&bc->ksStart);
			globalEnd.Set(&bc->ksFinish);
			
			totalSpace.Set(&globalEnd);
			totalSpace.Sub(&globalStart);
			totalSpace.AddOne();
			
			Int gpusInt;
			gpusInt.SetInt32(numGPUs);
			chunkSpace.Set(&totalSpace);
			chunkSpace.Div(&gpusInt);
			
			// Offset this specific GPU's starting point
			local_ksStart.Set(&chunkSpace);
			local_ksStart.Mult(thId);
			local_ksStart.Add(&globalStart);
			
			// Set this specific GPU's end point
			if (thId == numGPUs - 1) {
				local_ksFinish.Set(&globalEnd);
			} else {
				local_ksFinish.Set(&local_ksStart);
				local_ksFinish.Add(&chunkSpace);
				local_ksFinish.SubOne(); // Prevent overlap with the next GPU
			}

		taskSize.Set(&local_ksFinish);     
		taskSize.Sub(&local_ksStart);      
		taskSize.AddOne();
		stepThread.Set(&taskSize);
		stepThread.Div(&numthread);

		// Initialize GPU geometry with the isolated local anchors
		getGPUStartingKeys(local_ksStart, local_ksFinish, g.GetGroupSize(), numThreadsGPU, publicKeys, (uint64_t)(1ULL * local_idxcount * g.GetStepSize()));
		ok = g.SetKeys(publicKeys);
		needsNewBlock = false;

		// Calculate static upper bits for SEP dual-filter
		if (scConfig != NULL && (scConfig->useSEP || scConfig->enabled)) {
			// Calculate static HD for the upper geographic bits
			upper_hd = __builtin_popcountll(local_ksStart.bits64[1] ^ scConfig->rawTarget[1]) +
			           __builtin_popcountll(local_ksStart.bits64[2] ^ scConfig->rawTarget[2]) +
			           __builtin_popcountll(local_ksStart.bits64[3] ^ scConfig->rawTarget[3]);
			
			// Calculate static absolute density for the upper geographic bits
			upper_abs_pop = __builtin_popcountll(local_ksStart.bits64[1]) +
			                __builtin_popcountll(local_ksStart.bits64[2]) +
			                __builtin_popcountll(local_ksStart.bits64[3]);
		}

		// For Single/Multi-GPU Vanilla: Keep trackers aligned with local bounds
		thread_currentSeed.Set(&local_ksStart);
		thread_limitSeed.Set(&local_ksFinish);
		
		// PRINT THE SLICE
		printf("[Vanilla Engine] GPU %d Workload: %s to %s\n", 
			thId, local_ksStart.GetBase16().c_str(), local_ksFinish.GetBase16().c_str());
	}

	ttot = Timer::get_tick() - t0;
	printf("Initialization completed in %.2f seconds \n", ttot);
	fflush(stdout);

	ph->hasStarted = true;
	printf("GPU Started ! \r");
	fflush(stdout);

	t0 = Timer::get_tick();
	endOfSearch = false;

	// Hybrid Engine Bit Expander (CPU Side Only)
	auto expand_seed = [&](Int& seed, Int& key, const XorMask256& xor_mask) {
		key.SetInt32(0);
		// 256-BIT NATIVE XOR INJECTION: Flips weak bits anywhere in the keyspace
		key.bits64[0] = scConfig->lockVals[0] ^ xor_mask.m[0]; 
		key.bits64[1] = scConfig->lockVals[1] ^ xor_mask.m[1];
		key.bits64[2] = scConfig->lockVals[2] ^ xor_mask.m[2];
		key.bits64[3] = scConfig->lockVals[3] ^ xor_mask.m[3];

		for (int fb = 0; fb < scConfig->numFreeBits; fb++) {
			int pos = scConfig->freeBitPositions[fb];
			int seedLimb = fb >> 6;
			int seedBit = fb & 63;
			
			if ((seed.bits64[seedLimb] >> seedBit) & 1ULL) {
				key.bits64[pos >> 6] |= (1ULL << (pos & 63));
			}
		}
	};

	while (ok && !endOfSearch) {

		if (!Pause) {	
			
			// ==========================================
			// HYBRID BLOCK GENERATOR
			// ==========================================
			if (needsNewBlock && useStringCrack) {
				if (thread_currentSeed.IsGreaterOrEqual(&thread_limitSeed)) {
					endOfSearch = true;
					break;
				}

				Int mask;
				mask.SetInt32(1);
				mask.ShiftL(sc_lowerFreeBitsCount);
				mask.Sub(1);

				// Lock the exact end boundary of this block
				// (thread_blockEndSeed is now declared at function level)
				thread_blockEndSeed.Set(&thread_currentSeed);
				thread_blockEndSeed.ShiftR(sc_lowerFreeBitsCount);
				thread_blockEndSeed.ShiftL(sc_lowerFreeBitsCount);
				thread_blockEndSeed.Add(&mask);

				if (thread_blockEndSeed.IsGreaterOrEqual(&thread_limitSeed)) {
					thread_blockEndSeed.Set(&thread_limitSeed);
					thread_blockEndSeed.Sub(1);
				}

				Int ksStart, ksFinish;
				expand_seed(thread_currentSeed, ksStart, xor_masks[current_mutation]);
				expand_seed(thread_blockEndSeed, ksFinish, xor_masks[current_mutation]);

				// FIX: Update local state, DO NOT touch global bc
				local_ksStart.Set(&ksStart);
				local_ksFinish.Set(&ksFinish);

				taskSize.Set(&local_ksFinish);
				taskSize.Sub(&local_ksStart);
				taskSize.AddOne();

				stepThread.Set(&taskSize);
				stepThread.Div(&numthread);

				if (isFirstBlock) {
					// First block ONLY: Build geometry from scratch
					getGPUStartingKeys(local_ksStart, local_ksFinish, g.GetGroupSize(), numThreadsGPU, publicKeys, 0);
					isFirstBlock = false;
				} else {
					// All subsequent blocks: Mathematically teleport
					Int deltaScalar;
					deltaScalar.Set(&local_ksStart); // FIX
					
					// Failsafe for elliptic curve boundaries
					if (deltaScalar.IsLower(&previous_ksStart)) {
						deltaScalar.Add(&secp->order);
					}
					deltaScalar.Sub(&previous_ksStart);

					TeleportGrid(publicKeys, numThreadsGPU, deltaScalar);
				}

				// Store current anchor for the next jump calculation
				previous_ksStart.Set(&local_ksStart); // FIX

				ok = g.SetKeys(publicKeys);

				local_idxcount = 0; // FIX
				keycount.SetInt32(0);
				needsNewBlock = false;
			}

			if (randomMode && !useStringCrack) {
				RandomJump_K_last.Set(&RandomJump_K);
				RandomJump_K_tot.Add(&RandomJump_K);
				RandomJump_K.Rand(256);
				RandomJump_K.Mod(&stepThread);
				RandomJump_K.Sub(&RandomJump_K_tot);
				
				if (RandomJump_K.IsNegative()) {
					RandomJump_K.Neg();
					RandomJump_P = secp->ComputePublicKey(&RandomJump_K);
					RandomJump_P.y.ModNeg();
					RandomJump_K.Neg();
				} else {
					RandomJump_P = secp->ComputePublicKey(&RandomJump_K);
				}
				ok = g.SetRandomJump(RandomJump_P);
			}

			// ==========================================
			// THE MUSCLE: Standard Kernel Launch
			// ==========================================
			uint64_t step_thread_lo = stepThread.bits64[0];
			uint64_t ks_start_lo = local_ksStart.bits64[0];
			ok = g.Launch(found, true, ks_start_lo, step_thread_lo, local_idxcount, upper_hd, upper_abs_pop);
			local_idxcount += 1; // FIX

			if (!randomMode && local_idxcount % 60 == 0) {
				saveBackup(local_idxcount, ttot, ph->gpuId); // FIX
			}

			ttot = Timer::get_tick() - t0 + t_Paused;

			keycount.SetInt32(local_idxcount - 1); // FIX
			keycount.Mult(STEP_SIZE);

			// ==========================================
			// RECONSTRUCTION (Linear and Flawless)
			// ==========================================
			for (int i = 0; i < (int)found.size() && !endOfSearch; i++) {
				ITEM it = found[i];

				part_key.Set(&stepThread);
				part_key.Mult(it.thId);

				// FIX: Reconstruct using the thread's local anchor!
				privkey.Set(&local_ksStart);
				privkey.Add(&part_key);
				
				if (randomMode && !useStringCrack) {
					privkey.Add(&RandomJump_K_tot);
					privkey.Sub(&RandomJump_K_last);
				} else {
					privkey.Add(&keycount);
				}
				
				checkAddr(*(address_t*)(it.hash), it.hash, privkey, it.incr, it.endo, it.mode);
			}

			keycount.Add(STEP_SIZE);
			keycount.Mult(numThreadsGPU);

			if (useStringCrack) {
				sc_keys_n += (1ULL * STEP_SIZE * numThreadsGPU);

				if (keycount.IsGreaterOrEqual(&taskSize)) {
					needsNewBlock = true;
					
					// SHIFT REALITY: Check the same seed block against the next HD permutation
					current_mutation++;
					if (current_mutation >= xor_masks.size()) {
						current_mutation = 0; // Reset mutations
						
						// THE FIX: Snap exactly to the start of the next block. DO NOT OVERSHOOT.
						thread_currentSeed.Set(&thread_blockEndSeed);
						thread_currentSeed.AddOne();
					}
				}
			} else {
				if (keycount.IsGreaterOrEqual(&taskSize)) {
					needsNewBlock = true;
				}
			}

		} else {
			printf("Pausing...\r");
			fflush(stdout);
			g.FreeGPUEngine();
			Paused = true;
			t_Paused = ttot;
		}
		
		// Stats Output
		if (useStringCrack) {
			
			// Push local keys to the global array so Thread 0 can see them
			counters[thId] = sc_keys_n;

			// ONLY Thread 0 is allowed to print to prevent console garbling
			if (thId == 0) {
				uint64_t total_cluster_keys = 0;
				for (int i = 0; i < numGPUs; i++) {
					total_cluster_keys += counters[i];
				}

				// UI Throttle: Only print every 0.5 seconds
				if (ttot - sc_last_print_time >= 0.5) {
					// FIX: Pass sc_last_print_time instead of tprev
					PrintStatsStringCrack(total_cluster_keys, sc_keys_n_prev, ttot, sc_last_print_time,
						thread_currentSeed, thread_limitSeed, scConfig->seedOffsetInt,
						scConfig->numLockedBits, nbFoundKey, 0.0);
					sc_keys_n_prev = total_cluster_keys;
					sc_last_print_time = ttot;
				}
			}
			
			if (thread_currentSeed.IsGreaterOrEqual(&thread_limitSeed)) {
				if (thId == 0) {
					uint64_t total_cluster_keys = 0;
					for (int i = 0; i < numGPUs; i++) {
						total_cluster_keys += counters[i];
					}
					// FIX: Pass sc_last_print_time instead of tprev
					PrintStatsStringCrack(total_cluster_keys, sc_keys_n_prev, ttot, sc_last_print_time,
						thread_currentSeed, thread_limitSeed, scConfig->seedOffsetInt,
						scConfig->numLockedBits, nbFoundKey, 0.0);

					double avg_speed = static_cast<double>(total_cluster_keys) / (ttot * 1000000.0);
					printf("\n[Hybrid Engine] Cluster Finished! Avg: %.1f [MK/s] - Found: %d\n", avg_speed, nbFoundKey);
					fflush(stdout);
				}
				// All threads exit when their specific chunk is done
				endOfSearch = true; 
			}
		} else {
			// 1. Increment the local key counter
			keys_n += (1ULL * STEP_SIZE * numThreadsGPU);
			
			// 2. Push local keys to the global array so Thread 0 can see them
			counters[thId] = keys_n;

			// 3. ONLY Thread 0 is allowed to print to prevent console garbling
			if (thId == 0) {
				uint64_t total_cluster_keys = 0;
				for (int i = 0; i < numGPUs; i++) {
					total_cluster_keys += counters[i];
				}

				// UI Throttle: Only print every 0.5 seconds (prevents terminal lag)
				if (ttot - sc_last_print_time >= 0.5) {
					PrintStats(total_cluster_keys, sc_keys_n_prev, ttot, sc_last_print_time, taskSize, keycount);
					sc_keys_n_prev = total_cluster_keys;
					sc_last_print_time = ttot;
				}
			}

			if (keycount.IsGreaterOrEqual(&taskSize) && (!randomMode)) {
				// Only Thread 0 prints the final summary
				if (thId == 0) {
					uint64_t total_cluster_keys = 0;
					for (int i = 0; i < numGPUs; i++) {
						total_cluster_keys += counters[i];
					}
					double avg_speed = static_cast<double>(total_cluster_keys) / (ttot * 1000000.0);
					printf("\nRange Finished! - Average Speed: %.1f [MK/s] - Found: %d   \r\n", avg_speed, nbFoundKey);
					fflush(stdout);
				}
				endOfSearch = true;
			}
		}

		tprev = ttot;
	}

	ph->isRunning = false;
	endOfSearch = true;
	delete[] publicKeys; 
}

void VanitySearch::PrintStatsStringCrack(
    uint64_t keys_n, uint64_t keys_n_prev, 
    double ttot, double tprev, 
    Int& seedsScanned, Int& seedCount, Int& seedStart,
    int numLockedBits, int nbFound,
    double realTimeSpeed) 
{
	double speed;
	double perc;
	double bkeys;
	
	// Calculate progress as: (current - start) / (end - start) * 100
	Int distance;
	distance.Set(&seedsScanned);
	distance.Sub(&seedStart);
	
	Int totalRange;
	totalRange.Set(&seedCount);
	totalRange.Sub(&seedStart);
	
	// Use 128-bit arithmetic for accurate percentage
	const double TWO_TO_64 = 18446744073709551616.0;
	
	double distanceHi = (double)distance.bits64[1] * TWO_TO_64;
	double distanceLo = (double)distance.bits64[0];
	double distanceFull = distanceHi + distanceLo;
	
	double rangeHi = (double)totalRange.bits64[1] * TWO_TO_64;
	double rangeLo = (double)totalRange.bits64[0];
	double rangeFull = rangeHi + rangeLo;
	
	if (rangeFull > 0.0) {
		perc = distanceFull / rangeFull * 100.0;
	} else {
		perc = 0.0;
	}

	// 1. Get the Raw Seed Processing Speed
	if (realTimeSpeed > 0.0) {
		speed = realTimeSpeed;
	} else if (ttot > tprev) {
		// FIX: Use true instantaneous speed calculation instead of total average
		speed = (double)(keys_n - keys_n_prev) / ((ttot - tprev) * 1000000.0);
	} else {
		speed = 0.0;
	}

	// =========================================================================
	// 2. NEW: CALCULATE EFFECTIVE SPEED (Ignoring Geographic Bits 64-70)
	// =========================================================================
	int free_under_64 = 0;
	int puzzle_bits_under_64 = 64; // Default to 64
	
	if (scConfig != NULL) {
		// FIX: Bound the calculation if the search space is explicitly smaller than 64 bits
		if (scConfig->puzzleBits > 0 && scConfig->puzzleBits < 64) {
			puzzle_bits_under_64 = scConfig->puzzleBits;
		}
		// Count how many "Free Bits" are located in the relevant range
		for(int i = 0; i < scConfig->numFreeBits; i++) {
			if(scConfig->freeBitPositions[i] < puzzle_bits_under_64) {
				free_under_64++;
			}
		}
	}
	
	// The algorithmic multiplier is purely based on explicitly locked bits in that range
	int algo_locks = puzzle_bits_under_64 - free_under_64;
	if (algo_locks < 0) algo_locks = 0;
	
	// Effective Speed = Raw Speed * 2^(Algorithmic Locks)
	double eff_speed = speed * pow(2.0, algo_locks);
	// =========================================================================

	bkeys = (double)keys_n / 1000000000.0;

	// Format: trim hex to actual bit length
	int seedBitLen = seedsScanned.GetBitLength();
	std::string seedHex = seedsScanned.GetBase16();
	if ((int)seedHex.length() > (seedBitLen + 3) / 4) {
		seedHex = seedHex.substr(seedHex.length() - (seedBitLen + 3) / 4);
	}
	
	int countBitLen = seedCount.GetBitLength();
	std::string countHex = seedCount.GetBase16();
	if ((int)countHex.length() > (countBitLen + 3) / 4) {
		countHex = countHex.substr(countHex.length() - (countBitLen + 3) / 4);
	}

	// 3. Print both Raw and Effective Speed!
	printf("Raw: %.1f MK/s | Eff: %.1f MK/s - %.2f BKeys - %s/%s [%.2f%%] - Found: %d     \r",
		speed, eff_speed, bkeys,
		seedHex.c_str(),
		countHex.c_str(),
		perc, nbFound);

	fflush(stdout);
}


void VanitySearch::PrintStats(uint64_t keys_n, uint64_t keys_n_prev, double ttot, double tprev, Int taskSize, Int keycount) {

	double speed;
	double perc;
	double log_keys;
	double bkeys;

	Int Perc;

	Perc.Set(&taskSize);
	Perc.Mult(65536);
	Perc.Div(&keycount);


	if (ttot > tprev) {
		// Instantaneous speed: (Current Keys - Previous Keys) / Time Elapsed
		speed = (double)(keys_n - keys_n_prev) / ((ttot - tprev) * 1000000.0);
	} else {
		speed = 0.0;
	}


	perc = (double)(1 / Perc.ToDouble()*100*65536);


	log_keys = log2(static_cast<double>(keys_n));
	bkeys = static_cast<double>(keys_n);
	bkeys = bkeys / 1000000000;

	int h_run = static_cast<int32_t>(ttot) / 3600;
	int m_run = (static_cast<int32_t>(ttot) % 3600) / 60;
	int s_run = static_cast<int32_t>(ttot) % 60;
	int d_run = static_cast<int32_t>(ttot * 10) % 10;

	double tempo_tot_stimato = ttot / (perc / 100.0);
	double end_tt = tempo_tot_stimato - ttot;

	int h_end = static_cast<int32_t>(end_tt) / 3600;
	int m_end = (static_cast<int32_t>(end_tt) % 3600) / 60;
	int s_end = static_cast<int32_t>(end_tt) % 60;
	int d_end = static_cast<int32_t>(end_tt * 10) % 10;



	if (randomMode) {
		if (!Paused) {

			printf("%.1f MK/s - %.0f BKeys - 2^%.2f [%.2f%%] - RUN: %02d:%02d:%02d.%01d - Found: %d     ",
				speed, bkeys, log_keys, perc, h_run, m_run, s_run, d_run, nbFoundKey);

		}
		else {
			printf("Paused - %.0f Bkeys -  2^%.2f [%.2f%%] - RUN: %02d:%02d:%02d.%01d - Found: %d     ",
				bkeys, log_keys, perc, h_run, m_run, s_run, d_run, nbFoundKey);

			endOfSearch = true;
		}
	}
	else {
		if (!Paused) {

			if (h_end >= 0)
				printf("%.1f MK/s - %.0f BKeys - 2^%.2f [%.2f%%] - RUN: %02d:%02d:%02d.%01d|END: %02d:%02d:%02d.%01d - Found: %d     ",
					speed, bkeys, log_keys, perc, h_run, m_run, s_run, d_run, h_end, m_end, s_end, d_end, nbFoundKey);
			else
				printf("%.1f MK/s - %.0f BKeys - 2^%.2f [%.2f%%] - RUN: %02d:%02d:%02d.%01d|END: Too much bro - Found: %d     ",
					speed, bkeys, log_keys, perc, h_run, m_run, s_run, d_run, nbFoundKey);
		}
		else {
			printf("Paused - %.0f BKeys - 2^%.2f [%.2f%%] - RUN: %02d:%02d:%02d.%01d|END: %02d:%02d:%02d.%01d - Found: %d     ",
				bkeys,log_keys, perc, h_run, m_run, s_run, d_run, h_end, m_end, s_end, d_end, nbFoundKey);

			endOfSearch = true;
		}
	}


	printf("\r");


	fflush(stdout);
}


void VanitySearch::saveBackup(int idxcount, double t_Paused, int gpuid) {
	std::string filename = "VSbackup_gpu" + std::to_string(gpuid) + ".dat";
	std::ofstream outFile(filename, std::ios::binary);
	if (outFile) {
		outFile.write(reinterpret_cast<const char*>(&idxcount), sizeof(idxcount));
		outFile.write(reinterpret_cast<const char*>(&t_Paused), sizeof(t_Paused));
		outFile.close();
	}
	else {
		std::cerr << "Error opening file for writing: " << filename << "\n";
	}
}

bool VanitySearch::isAlive(TH_PARAM * p) {

	bool isAlive = true;
	int total = numGPUs;
	for (int i = 0; i < total; i++)
		isAlive = isAlive && p[i].isRunning;

	return isAlive;
}

bool VanitySearch::hasStarted(TH_PARAM * p) {

	bool hasStarted = true;
	int total = numGPUs;
	for (int i = 0; i < total; i++)
		hasStarted = hasStarted && p[i].hasStarted;

	return hasStarted;
}

uint64_t VanitySearch::getGPUCount() {

	uint64_t count = 0;
	for (int i = 0; i < numGPUs; i++) {
		count += counters[i];
	}
	return count;
}

void VanitySearch::saveProgress(TH_PARAM* p, Int& lastSaveKey, BITCRACK_PARAM* bc) {

	Int lowerKey;
	lowerKey.Set(&p[0].THnextKey);

	int total = numGPUs;
	for (int i = 0; i < total; i++) {
		if (p[i].THnextKey.IsLower(&lowerKey))
			lowerKey.Set(&p[i].THnextKey);
	}

	if (lowerKey.IsLowerOrEqual(&lastSaveKey)) return;
	lastSaveKey.Set(&lowerKey);
}

void VanitySearch::Search(std::vector<int> gpuId, std::vector<int> gridSize) {

	//double t0;
	//double t1;
	endOfSearch = false;
	numGPUs = ((int)gpuId.size());
	nbFoundKey = 0;

	memset(counters, 0, sizeof(counters));	

	TH_PARAM* params = (TH_PARAM*)malloc(numGPUs * sizeof(TH_PARAM));
	memset(params, 0, numGPUs * sizeof(TH_PARAM));
	
	std::thread* threads = new std::thread[numGPUs];

#ifdef WIN64
	ghMutex = CreateMutex(NULL, FALSE, NULL);
	mutex = CreateMutex(NULL, FALSE, NULL);
#else
	ghMutex = PTHREAD_MUTEX_INITIALIZER;
	mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

	// Launch GPU threads
	for (int i = 0; i < numGPUs; i++) {
		params[i].obj = this;
		params[i].threadId = i;
		params[i].isRunning = true;
		params[i].gpuId = gpuId[i];
		params[i].smMultiplier = this->smMultiplier;
		params[i].gridSizeX = gridSize[i];
		params[i].gridSizeY = gridSize[i+1];
		params[i].THnextKey.Set(&bc->ksNext);
		
		threads[i] = std::thread(_FindKeyGPU, params + i);
	}

	while (!hasStarted(params)) {
		Timer::SleepMillis(500);
	}

	while (!endOfSearch) {
		Timer::SleepMillis(100);
	}

	
	if (params != nullptr) {
		free(params);
	}

}

string VanitySearch::GetHex(vector<unsigned char> &buffer) {

	string ret;

	char tmp[128];
	for (int i = 0; i < (int)buffer.size(); i++) {
		sprintf(tmp, "%02hhX", buffer[i]);
		ret.append(tmp);
	}

	return ret;
}

void VanitySearch::TeleportGrid(Point* p, int numThreads, Int& deltaScalar) {
    // 1. Calculate the single Jump Point
    Point P_delta = secp->ComputePublicKey(&deltaScalar);

    Int* dx = new Int[numThreads];
    Int* subp = new Int[numThreads];

    // 2. Calculate dx_i for all points
    for (int i = 0; i < numThreads; i++) {
        dx[i].ModSub(&P_delta.x, &p[i].x);
    }

    // 3. Cumulative product for Montgomery batch inversion
    subp[0].Set(&dx[0]);
    for (int i = 1; i < numThreads; i++) {
        subp[i].ModMulK1(&subp[i - 1], &dx[i]);
    }

    // 4. Invert the total product
    Int inverse;
    inverse.Set(&subp[numThreads - 1]);
    inverse.ModInv();

    // 5. Backtrack to find individual inverses
    Int newValue;
    for (int i = numThreads - 1; i > 0; i--) {
        newValue.ModMulK1(&subp[i - 1], &inverse); // inverse of dx[i]
        inverse.ModMulK1(&dx[i]);                  // pass inverse down to next
        dx[i].Set(&newValue);
    }
    dx[0].Set(&inverse); // dx[0] is now its own inverse

    // 6. Apply the EC addition formulas using the inverses
    Int lambda, lambdaSq, dy, x_new, y_new;
    for (int i = 0; i < numThreads; i++) {
        // lambda = (y_delta - y_i) * dx_i^-1
        dy.ModSub(&P_delta.y, &p[i].y);
        lambda.ModMulK1(&dy, &dx[i]);

        // lambdaSq = lambda^2
        lambdaSq.ModSquareK1(&lambda);

        // x_new = lambdaSq - x_i - x_delta
        x_new.ModSub(&lambdaSq, &p[i].x);
        x_new.ModSub(&P_delta.x);

        // y_new = lambda * (x_i - x_new) - y_i
        y_new.ModSub(&p[i].x, &x_new);
        y_new.ModMulK1(&lambda);
        y_new.ModSub(&p[i].y);

        // Update the point in place
        p[i].x.Set(&x_new);
        p[i].y.Set(&y_new);
    }

    delete[] dx;
    delete[] subp;
}
